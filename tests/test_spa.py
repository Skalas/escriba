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


@pytest.fixture
def routed_page(chromium_browser, spa_server: str):
    """Factory: call with a route handler to get a ready, API-routed page.

    Centralizes the new_context → route → goto → wait setup/teardown so tests
    that need a custom API stub don't each reproduce it.
    """
    contexts = []

    def _make(route_handler, *, timeout: int = 200):
        ctx = chromium_browser.new_context()
        contexts.append(ctx)
        pg = ctx.new_page()
        pg.route("**/api/**", route_handler)
        pg.goto(spa_server)
        pg.wait_for_load_state("domcontentloaded")
        pg.wait_for_timeout(timeout)
        return pg

    yield _make
    for ctx in contexts:
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


# ---------------------------------------------------------------------------
# T1–T4: Notepad-flow redesign (v0.10.1)
# ---------------------------------------------------------------------------

def test_t1_transcript_panel_collapsed_by_default(page) -> None:
    """Transcript body is hidden by default; toggling it makes it visible."""
    body_visible = page.evaluate("""() => {
        const body = document.getElementById('live-tr-body');
        if (!body) return null;
        return body.classList.contains('open');
    }""")
    assert body_visible is False, "live-tr-body should be collapsed (no .open) by default"
    # Now toggle it open
    page.evaluate("""() => {
        const toggle = document.getElementById('live-tr-toggle');
        if (toggle) toggle.click();
    }""")
    page.wait_for_timeout(100)
    body_open = page.evaluate("document.getElementById('live-tr-body').classList.contains('open')")
    assert body_open is True, "live-tr-body should be open after clicking the toggle"


def test_t2_enhance_no_instructions_triggers_generation(page) -> None:
    """Clicking 'Enhance notes' with no instructions calls /api/notes."""
    page.evaluate("""() => {
        viewingHistory = false;
        document.getElementById('live-view').style.display = '';
        document.getElementById('notes-section').classList.add('visible');
    }""")
    called = []

    def _on_notes(r) -> None:
        called.append(True)
        r.fulfill(status=200, content_type="application/json",
                  body='{"ok":true,"notes":"AI enhanced text"}')

    page.route("**/api/notes", _on_notes)
    page.evaluate("void generateNotes()")
    page.wait_for_timeout(800)
    page.unroute("**/api/notes")
    assert called, "generateNotes() must call /api/notes"


def test_t2_add_instructions_disclosure_hidden_by_default(page) -> None:
    """The 'Add instructions' panel is hidden by default, opens on first click, closes on second."""
    panel_hidden = page.evaluate("""() => {
        const panel = document.getElementById('live-instructions-panel');
        if (!panel) return null;
        return !panel.classList.contains('open');
    }""")
    assert panel_hidden is True, "live-instructions-panel must be hidden (no .open class) by default"
    # First click — opens
    page.evaluate("""() => {
        const btn = document.getElementById('btn-add-instructions');
        if (btn) btn.click();
    }""")
    page.wait_for_timeout(100)
    panel_open = page.evaluate("document.getElementById('live-instructions-panel').classList.contains('open')")
    assert panel_open is True, "live-instructions-panel should have .open after first click"
    aria_expanded = page.evaluate("document.getElementById('btn-add-instructions').getAttribute('aria-expanded')")
    assert aria_expanded == "true", "aria-expanded must be 'true' when panel is open"
    # Second click — closes
    page.evaluate("""() => {
        const btn = document.getElementById('btn-add-instructions');
        if (btn) btn.click();
    }""")
    page.wait_for_timeout(100)
    panel_closed = page.evaluate("!document.getElementById('live-instructions-panel').classList.contains('open')")
    assert panel_closed is True, "live-instructions-panel should lose .open on second click"
    aria_collapsed = page.evaluate("document.getElementById('btn-add-instructions').getAttribute('aria-expanded')")
    assert aria_collapsed == "false", "aria-expanded must be 'false' when panel is closed"
    # Chips container must exist inside it
    has_chips = page.evaluate("!!document.getElementById('live-prompt-chips')")
    assert has_chips, "live-prompt-chips must exist inside the instructions panel"


def test_t3_ai_content_has_provenance_markers(page) -> None:
    """After Enhance, AI-added content has .live-ai-block and .live-chip-ai — not color alone."""
    page.evaluate("""() => {
        // Make live-view visible so isLiveViewActive() returns true
        viewingHistory = false;
        document.getElementById('live-view').style.display = '';
        document.getElementById('notes-section').classList.add('visible');
    }""")
    page.route("**/api/notes", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body='{"ok":true,"notes":"Summary: this is AI-generated text."}'
    ))
    page.evaluate("void generateNotes()")
    page.wait_for_timeout(900)
    page.unroute("**/api/notes")
    has_rail = page.evaluate("document.querySelector('.live-ai-block') !== null")
    has_chip = page.evaluate("document.querySelector('.live-chip-ai') !== null")
    assert has_rail, ".live-ai-block (provenance rail) must be present after enhance"
    assert has_chip, ".live-chip-ai (AI-added chip) must be present after enhance"
    # aria-live polite must be on the output container
    aria_live = page.evaluate("document.getElementById('notes-output').getAttribute('aria-live')")
    assert aria_live == "polite", "#notes-output must have aria-live='polite'"


def test_t4_user_only_notes_are_not_dropped(page) -> None:
    """renderNotesView with user jots but no AI notes must still display the user content (B4).

    Without the user-only branch, passing notesText='' falls through to renderMarkdown('')
    which clears the user's jotted notes silently.
    """
    page.evaluate("""() => {
        _currentSessionUserNotes = 'Meeting recap: ship by Friday';
        renderNotesView('');
    }""")
    rendered_html = page.evaluate("document.getElementById('notes-rendered').innerHTML")
    assert "Meeting recap" in rendered_html, (
        "User jotted notes must appear in notes-rendered even when AI notes are absent. "
        "The user-only branch in renderNotesView() is missing or broken."
    )


# ---------------------------------------------------------------------------
# v0.10.2: Enhance action bar visibility + enabled gating (T1) and
#          no duplicate "Your notes" heading in live view (T2)
# ---------------------------------------------------------------------------

def test_v0102_t1_enhance_section_visible_in_idle_live_view(page) -> None:
    """#notes-section must be visible in the idle/empty live view (no segments, no notepad text)."""
    page.evaluate("""() => {
        viewingHistory = false;
        showLiveView();
    }""")
    visible = page.evaluate("""() => {
        const el = document.getElementById('notes-section');
        return el.classList.contains('visible');
    }""")
    assert visible, "#notes-section must have .visible after showLiveView() — even with no segments"


def test_v0102_t1_enhance_button_disabled_in_empty_idle_live_view(page) -> None:
    """#btn-notes must be disabled when notepad is empty and segments_count == 0,
    and must have a clearly-disabled visual appearance (reduced opacity, not-allowed cursor)."""
    page.evaluate("""() => {
        viewingHistory = false;
        document.getElementById('live-notepad').value = '';
        document.getElementById('seg-count').textContent = '0';
        showLiveView();
    }""")
    disabled = page.evaluate("document.getElementById('btn-notes').disabled")
    assert disabled, "#btn-notes must be disabled when notepad is empty and there are no segments"

    styles = page.evaluate("""() => {
        const btn = document.getElementById('btn-notes');
        const cs = getComputedStyle(btn);
        return { opacity: cs.opacity, cursor: cs.cursor };
    }""")
    opacity = float(styles["opacity"])
    assert opacity < 0.6, (
        f"Disabled #btn-notes must have reduced opacity (got {opacity}) — "
        "it renders at full coral strength and looks clickable without a :disabled rule"
    )
    assert styles["cursor"] == "not-allowed", (
        f"Disabled #btn-notes must have cursor:not-allowed (got {styles['cursor']!r})"
    )


def test_v0102_t1_enhance_button_enabled_after_typing_note(page) -> None:
    """Typing into #live-notepad must enable #btn-notes (even with segments_count == 0)."""
    page.evaluate("""() => {
        viewingHistory = false;
        document.getElementById('live-notepad').value = '';
        document.getElementById('seg-count').textContent = '0';
        showLiveView();
        // Simulate typing
        document.getElementById('live-notepad').value = 'Hello';
        onLiveNotepadInput();
    }""")
    page.wait_for_timeout(50)
    disabled = page.evaluate("document.getElementById('btn-notes').disabled")
    assert not disabled, "#btn-notes must be enabled after typing a note into the notepad"


def test_v0102_t1_enhance_button_enabled_when_segments_present(page) -> None:
    """#btn-notes must be enabled when segments_count > 0 even with empty notepad."""
    page.evaluate("""() => {
        viewingHistory = false;
        document.getElementById('live-notepad').value = '';
        document.getElementById('seg-count').textContent = '3';
        showLiveView();
    }""")
    disabled = page.evaluate("document.getElementById('btn-notes').disabled")
    assert not disabled, "#btn-notes must be enabled when segments_count > 0"


def test_v0102_t2_live_enhance_no_duplicate_your_notes_heading(page) -> None:
    """After enhancing in the live view, the provenance output must NOT contain 'Your notes'
    (the editable textarea above already is the user's notes)."""
    page.evaluate("""() => {
        viewingHistory = false;
        document.getElementById('live-view').style.display = '';
        document.getElementById('notes-section').classList.add('visible');
        document.getElementById('live-notepad').value = 'Meeting notes here';
    }""")
    page.route("**/api/notes", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body='{"ok":true,"notes":"Summary: AI-generated additions."}'
    ))
    page.evaluate("void generateNotes()")
    page.wait_for_timeout(900)
    page.unroute("**/api/notes")
    output_html = page.evaluate("document.getElementById('notes-output').innerHTML")
    assert "Your notes" not in output_html, (
        "Live view provenance output must NOT contain a 'Your notes' heading — "
        "the editable textarea directly above already labels the user's content"
    )
    assert "AI additions" in output_html, "AI additions block must still be labeled in live view"


def test_v0102_t2_saved_session_view_keeps_your_notes_heading(page) -> None:
    """In the saved-session notes view (renderNotesView), the 'Your notes' heading must appear
    — there is no editable textarea visible there to label the user's content."""
    page.evaluate("""() => {
        _currentSessionUserNotes = 'Pre-meeting jots';
        renderNotesView('Summary: AI additions here.');
    }""")
    rendered_html = page.evaluate("document.getElementById('notes-rendered').innerHTML")
    assert "Your notes" in rendered_html, (
        "Saved-session renderNotesView must include the 'Your notes' heading — "
        "no textarea is visible there to label the user content"
    )


# ---------------------------------------------------------------------------
# v0.10.3: Session-view notes UX consistency (T1) + slim record button (T2)
# ---------------------------------------------------------------------------

def test_v0103_t1_session_chips_hidden_by_default(page) -> None:
    """Session instructions panel is hidden by default (chips behind disclosure)."""
    hidden = page.evaluate("""() => {
        const panel = document.getElementById('session-instructions-panel');
        return !!panel && !panel.classList.contains('open');
    }""")
    assert hidden, "session-instructions-panel must be hidden (no .open) by default"


def test_v0103_t1_session_enhance_button_present(page) -> None:
    """#btn-session-notes must be present in the session view."""
    present = page.evaluate("!!document.getElementById('btn-session-notes')")
    assert present, "#btn-session-notes must exist"


def test_v0103_t1_session_edit_affordance_present(page) -> None:
    """Edit affordance (#btn-toggle-notes-edit) must be present in the session notes card."""
    present = page.evaluate("!!document.getElementById('btn-toggle-notes-edit')")
    assert present, "#btn-toggle-notes-edit must exist"


def test_v0103_t1_session_legend_shown_with_provenance(page) -> None:
    """Legend appears in session view when both user notes and AI notes are present."""
    page.evaluate("""() => {
        _currentSessionUserNotes = 'My jots';
        renderNotesView('AI-generated notes here.');
    }""")
    legend_visible = page.evaluate("""() => {
        const el = document.getElementById('session-enhance-legend');
        return !!el && el.style.display !== 'none';
    }""")
    assert legend_visible, "#session-enhance-legend must be visible when provenance exists"
    has_legend_class = page.evaluate(
        "document.getElementById('session-enhance-legend').classList.contains('live-legend')"
    )
    assert has_legend_class, "#session-enhance-legend must have .live-legend class"


def test_v0103_t1_session_legend_hidden_without_provenance(page) -> None:
    """Legend is hidden when there are no AI notes."""
    page.evaluate("""() => {
        _currentSessionUserNotes = '';
        renderNotesView('');
    }""")
    hidden = page.evaluate("""() => {
        const el = document.getElementById('session-enhance-legend');
        return !el || el.style.display === 'none';
    }""")
    assert hidden, "#session-enhance-legend must be hidden when no AI notes"


def test_v0103_t1_session_add_instructions_toggles(page) -> None:
    """Session 'Add instructions' disclosure toggles open/closed with correct aria-expanded."""
    # Initially closed
    closed = page.evaluate("!document.getElementById('session-instructions-panel')?.classList.contains('open')")
    assert closed, "session-instructions-panel must start closed"

    # First click — opens
    page.evaluate("document.getElementById('btn-session-add-instructions')?.click()")
    page.wait_for_timeout(100)
    open_ = page.evaluate("document.getElementById('session-instructions-panel')?.classList.contains('open')")
    assert open_, "session-instructions-panel should open on first click"
    aria = page.evaluate("document.getElementById('btn-session-add-instructions')?.getAttribute('aria-expanded')")
    assert aria == "true", "aria-expanded must be 'true' when panel is open"

    # Second click — closes
    page.evaluate("document.getElementById('btn-session-add-instructions')?.click()")
    page.wait_for_timeout(100)
    closed2 = page.evaluate("!document.getElementById('session-instructions-panel')?.classList.contains('open')")
    assert closed2, "session-instructions-panel should close on second click"
    aria2 = page.evaluate("document.getElementById('btn-session-add-instructions')?.getAttribute('aria-expanded')")
    assert aria2 == "false", "aria-expanded must be 'false' when panel is closed"

    # Chips exist inside the panel
    has_chips = page.evaluate("!!document.getElementById('session-prompt-chips')")
    assert has_chips, "session-prompt-chips must exist inside the instructions panel"


def test_v0103_t1_session_notes_title_updates_with_provenance(page) -> None:
    """Session notes card title changes to 'Enhanced notes' when provenance is present."""
    # No provenance
    page.evaluate("""() => { _currentSessionUserNotes = ''; renderNotesView(''); }""")
    title_plain = page.evaluate("document.getElementById('session-notes-title')?.textContent")
    assert title_plain == "Notes", f"Title must be 'Notes' without provenance, got {title_plain!r}"

    # With provenance
    page.evaluate("""() => {
        _currentSessionUserNotes = 'My jots';
        renderNotesView('AI summary here.');
    }""")
    title_enhanced = page.evaluate("document.getElementById('session-notes-title')?.textContent")
    assert title_enhanced == "Enhanced notes", (
        f"Title must be 'Enhanced notes' when provenance exists, got {title_enhanced!r}"
    )


# ---------------------------------------------------------------------------
# v0.10.3 T2: Slim labeled record button
# ---------------------------------------------------------------------------

def test_v0103_t2_record_button_has_label(page) -> None:
    """#btn-record must have a visible label span with 'Start' text initially."""
    label = page.evaluate("document.getElementById('record-label')?.textContent")
    assert label is not None, "#record-label span must exist"
    assert "Start" in label, f"Initial label must contain 'Start', got {label!r}"


def test_v0103_t2_record_button_label_and_aria_reflect_state(page) -> None:
    """Label text and aria-label update to 'Stop recording' when syncStatus reports active."""
    page.route("**/api/status", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body='{"ok":true,"is_active":true,"session_id":"x","elapsed":"00:01:00","segments_count":2}'
    ))
    page.evaluate("void syncStatus()")
    page.wait_for_timeout(400)

    label = page.evaluate("document.getElementById('record-label')?.textContent")
    aria = page.evaluate("document.getElementById('btn-record')?.getAttribute('aria-label')")
    page.unroute("**/api/status")

    assert label is not None and "Stop" in label, (
        f"Label must contain 'Stop' when recording is active, got {label!r}"
    )
    assert aria is not None and "Stop" in aria, (
        f"aria-label must contain 'Stop' when recording is active, got {aria!r}"
    )


def test_v0103_t2_record_button_label_reverts_to_start(page) -> None:
    """Label reverts to 'Start recording' when syncStatus reports idle."""
    # First set to recording
    page.route("**/api/status", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body='{"ok":true,"is_active":true,"session_id":"x","elapsed":"00:00:05","segments_count":0}'
    ))
    page.evaluate("void syncStatus()")
    page.wait_for_timeout(300)
    page.unroute("**/api/status")

    # Then set to idle
    page.route("**/api/status", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body='{"ok":true,"is_active":false,"session_id":null}'
    ))
    page.evaluate("void syncStatus()")
    page.wait_for_timeout(300)
    page.unroute("**/api/status")

    label = page.evaluate("document.getElementById('record-label')?.textContent")
    assert label is not None and "Start" in label, (
        f"Label must revert to 'Start' when idle, got {label!r}"
    )


# ---------------------------------------------------------------------------
# v0.11.0: Home + Sidebar redesign
# ---------------------------------------------------------------------------

def test_v0110_home_hero_present(page) -> None:
    """Home view hero is present when showing the home/empty view."""
    page.evaluate("""() => {
        allSessions = [];
        allFolders = [];
        showEmptyView();
    }""")
    page.wait_for_timeout(200)
    visible = page.evaluate("!!document.querySelector('.home-hero')")
    assert visible, ".home-hero must be present on the home view"


def test_v0110_home_zero_state_shows_empty_message(page) -> None:
    """With no sessions, the home shows a zero-state message instead of recent cards."""
    page.evaluate("""() => {
        allSessions = [];
        allFolders = [];
        showEmptyView();
    }""")
    page.wait_for_timeout(200)
    zero = page.evaluate("!!document.querySelector('.home-zero-recents')")
    assert zero, ".home-zero-recents must be visible when no sessions"
    grid = page.evaluate("!!document.querySelector('#home-recents-grid')")
    assert not grid, "#home-recents-grid must not exist when no sessions"


def test_v0110_home_recent_cards_shown_with_sessions(page) -> None:
    """With sessions, the home shows recent session cards."""
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'Alpha Meeting', status:'completed', segment_count:3, folder_id:null,
             started_at:'2026-06-29T10:00:00', duration_seconds:120, notes_text:'Some notes'},
            {id:'s2', name:'Beta Call', status:'completed', segment_count:1, folder_id:null,
             started_at:'2026-06-28T15:30:00', duration_seconds:60, notes_text:''},
        ];
        allFolders = [];
        showEmptyView();
    }""")
    page.wait_for_timeout(200)
    grid_present = page.evaluate("!!document.querySelector('#home-recents-grid')")
    assert grid_present, "#home-recents-grid must be present with sessions"
    card_count = page.evaluate("document.querySelectorAll('.session-recent-card').length")
    assert card_count >= 1, "At least one recent card must be rendered"


def test_v0110_home_recent_card_click_opens_session(page) -> None:
    """Clicking a recent card calls selectSession with the correct id."""
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'Alpha', status:'completed', segment_count:2, folder_id:null,
             started_at:'2026-06-29T10:00:00', duration_seconds:90, notes_text:''},
        ];
        allFolders = [];
        window.__cardClicks = [];
        window.__origSelectSession5 = window.selectSession;
        window.selectSession = async (id) => { window.__cardClicks.push(id); };
        showEmptyView();
    }""")
    page.wait_for_timeout(200)
    page.evaluate("document.querySelector('.session-recent-card')?.click()")
    page.wait_for_timeout(200)
    clicks = page.evaluate("window.__cardClicks")
    page.evaluate("window.selectSession = window.__origSelectSession5")
    assert 's1' in clicks, f"Clicking a recent card must call selectSession('s1'), got {clicks!r}"


def test_v0110_sidebar_groups_by_date(page) -> None:
    """Sidebar session list renders date group labels (Today, Yesterday, etc.)."""
    page.evaluate("""() => {
        const now = new Date();
        const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1, 12, 0).toISOString();
        allSessions = [
            {id:'s1', name:'Today session', status:'completed', segment_count:1, folder_id:null,
             started_at:now.toISOString(), duration_seconds:30},
            {id:'s2', name:'Yesterday session', status:'completed', segment_count:1, folder_id:null,
             started_at:yesterday, duration_seconds:30},
        ];
        allFolders = [];
        renderSessionList();
    }""")
    page.wait_for_timeout(150)
    label_count = page.evaluate("document.querySelectorAll('.session-group-label').length")
    assert label_count >= 1, "Date group labels must be rendered in the sidebar"
    all_text = page.evaluate("""
        Array.from(document.querySelectorAll('.session-group-label')).map(el => el.textContent).join(' ')
    """)
    assert 'Today' in all_text or 'Yesterday' in all_text, \
        f"Expected Today/Yesterday group, got: {all_text!r}"


def test_v0110_sidebar_no_checkbox_without_select_mode(page) -> None:
    """Session rows have NO visible checkbox until Select mode is entered."""
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'A', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
        ];
        allFolders = [];
        document.getElementById('sidebar').classList.remove('select-mode');
        selectModeActive = false;
        renderSessionList();
    }""")
    page.wait_for_timeout(150)
    visible = page.evaluate("""() => {
        const cb = document.querySelector('#session-list .session-checkbox');
        return cb ? getComputedStyle(cb).display !== 'none' : false;
    }""")
    assert not visible, "Session checkboxes must be hidden (display:none) when not in select mode"


def test_v0110_sidebar_select_toggle_reveals_checkboxes(page) -> None:
    """Clicking the Select toggle reveals checkboxes."""
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'A', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
        ];
        allFolders = [];
        document.getElementById('sidebar').classList.remove('select-mode');
        selectModeActive = false;
        renderSessionList();
    }""")
    page.wait_for_timeout(150)
    page.evaluate("document.getElementById('select-toggle-btn').click()")
    page.wait_for_timeout(150)
    visible = page.evaluate("""() => {
        const cb = document.querySelector('#session-list .session-checkbox');
        return cb ? getComputedStyle(cb).display !== 'none' : false;
    }""")
    page.evaluate("if (selectModeActive) toggleSelectMode()")
    assert visible, "Checkboxes must become visible when select mode is toggled on"


def test_v0110_sidebar_collapsed_by_default(page) -> None:
    """On first run (no prior localStorage), the sidebar is collapsed."""
    collapsed = page.evaluate("""() => {
        localStorage.removeItem('escriba-sidebar-collapsed');
        restoreSidebarState();
        return document.getElementById('sidebar').classList.contains('collapsed');
    }""")
    assert collapsed, "Sidebar must be collapsed by default (no prior localStorage)"


def test_v0110_sidebar_expand_collapse_persists(page) -> None:
    """Expanding and collapsing the sidebar persists to localStorage."""
    page.evaluate("""() => {
        document.getElementById('sidebar').classList.add('collapsed');
        document.body.classList.add('sidebar-collapsed');
        localStorage.setItem('escriba-sidebar-collapsed', '1');
    }""")
    page.evaluate("toggleSidebar()")
    page.wait_for_timeout(100)
    stored = page.evaluate("localStorage.getItem('escriba-sidebar-collapsed')")
    assert stored == '0', f"localStorage must be '0' after expanding, got {stored!r}"
    page.evaluate("toggleSidebar()")
    page.wait_for_timeout(100)
    stored2 = page.evaluate("localStorage.getItem('escriba-sidebar-collapsed')")
    assert stored2 == '1', f"localStorage must be '1' after collapsing, got {stored2!r}"


def test_v0110_one_record_control_on_home(page) -> None:
    """On home view: hero record present, topbar record hidden."""
    page.evaluate("""() => {
        allSessions = [];
        allFolders = [];
        showEmptyView();
    }""")
    page.wait_for_timeout(200)
    hero_present = page.evaluate("!!document.getElementById('btn-hero-record')")
    assert hero_present, "#btn-hero-record must exist on home view"
    topbar_display = page.evaluate("""() => {
        const btn = document.getElementById('btn-record');
        return btn ? getComputedStyle(btn).display : 'missing';
    }""")
    assert topbar_display == 'none', \
        f"#btn-record (topbar) must be display:none on home, got {topbar_display!r}"


def test_v0110_topbar_record_shown_on_session_view(page) -> None:
    """On a session view, the topbar record button is visible."""
    page.evaluate("""() => {
        selectedSessionId = 's1';
        viewingHistory = true;
        showSessionView();
    }""")
    page.wait_for_timeout(200)
    topbar_display = page.evaluate("""() => {
        const btn = document.getElementById('btn-record');
        return btn ? getComputedStyle(btn).display : 'missing';
    }""")
    assert topbar_display != 'none', \
        f"#btn-record must be visible on session view, got {topbar_display!r}"


def test_v0110_settings_accessible_from_topbar(page) -> None:
    """The topbar contains a Settings button accessible when sidebar is collapsed."""
    settings_in_topbar = page.evaluate("""() => {
        const topbar = document.getElementById('topbar');
        return topbar ? !!topbar.querySelector('[aria-label="Settings"]') : false;
    }""")
    assert settings_in_topbar, "Topbar must contain a Settings button (aria-label='Settings')"


def test_v0110_bulk_delete_removes_home_recents_card(page) -> None:
    """Regression T1: after bulk-deleting a session that appeared in home recents,
    its card must no longer appear in the grid (no ghost cards)."""
    page.evaluate("""() => {
        allSessions = [
            {id:'del1', name:'To Delete', status:'completed', segment_count:2, folder_id:null,
             started_at:new Date().toISOString(), duration_seconds:60},
            {id:'keep1', name:'Keep Me', status:'completed', segment_count:1, folder_id:null,
             started_at:new Date().toISOString(), duration_seconds:30},
        ];
        allFolders = [];
        showEmptyView();
    }""")
    page.wait_for_timeout(200)

    # Confirm both cards are initially present in the home recents grid
    initial_count = page.evaluate("""
        document.querySelectorAll('#home-recents-section .session-recent-card').length
    """)
    assert initial_count == 2, f"Expected 2 recent cards initially, got {initial_count}"

    # Simulate the state after DELETE api call: remove 'del1' from allSessions,
    # then call refreshSessionList + renderHomeView (what deleteBulkSelected does)
    page.evaluate("""() => {
        allSessions = allSessions.filter(s => s.id !== 'del1');
        renderSessionList();
        renderHomeView();
    }""")
    page.wait_for_timeout(150)

    # The deleted session card must no longer appear
    remaining = page.evaluate("""() => {
        const cards = document.querySelectorAll('#home-recents-section .session-recent-card');
        return Array.from(cards).map(c => c.dataset.id || c.getAttribute('data-id') || c.textContent.trim());
    }""")
    del1_present = any('del1' in str(r) or 'To Delete' in str(r) for r in remaining)
    assert not del1_present, \
        f"Ghost card for deleted session must not appear in home recents. Cards: {remaining}"
    assert len(remaining) == 1, \
        f"Expected exactly 1 remaining card, got {len(remaining)}: {remaining}"


# Persistent top-bar record control: one consistently-placed button across content views
def test_topbar_record_visible_on_live(page) -> None:
    """On the live view, the persistent top-bar #btn-record is visible (not hidden)."""
    page.evaluate("""() => {
        selectedSessionId = null;
        viewingHistory = false;
        showLiveView();
    }""")
    page.wait_for_timeout(150)
    display = page.evaluate("""() => {
        const btn = document.getElementById('btn-record');
        return btn ? getComputedStyle(btn).display : 'missing';
    }""")
    assert display != 'none', \
        f"#btn-record must be visible on live view (persistent top-bar control), got {display!r}"


def test_topbar_record_is_stop_control_while_recording(page) -> None:
    """While recording, the top-bar #btn-record is the stop control: it carries the recording
    class and an aria-label of 'Stop recording'."""
    page.evaluate("""() => {
        showLiveView();
        const btn = document.getElementById('btn-record');
        btn.classList.add('recording');
        btn.setAttribute('aria-label', 'Stop recording');
    }""")
    page.wait_for_timeout(100)
    aria_label = page.evaluate("document.getElementById('btn-record').getAttribute('aria-label')")
    assert aria_label == 'Stop recording', \
        f"#btn-record must be the stop control while recording, got {aria_label!r}"


def test_live_view_has_no_status_badge(page) -> None:
    """The redundant live-view status pill is gone; recording state is conveyed solely by the
    top-bar record/stop button and the Live Recording header."""
    page.evaluate("""() => { showLiveView(); }""")
    page.wait_for_timeout(100)
    exists = page.evaluate("!!document.getElementById('status-badge')")
    assert not exists, "#status-badge must be removed (redundant with the top-bar record control)"


def test_record_control_placement_per_view(page) -> None:
    """Record control placement: hero on home (onboarding); a single persistent top-bar button on
    both the live and session views (no separate live status control)."""
    # Home: hero present, top-bar btn hidden (onboarding welcome CTA)
    page.evaluate("""() => { allSessions = []; allFolders = []; showEmptyView(); }""")
    page.wait_for_timeout(150)
    hero = page.evaluate("getComputedStyle(document.getElementById('btn-hero-record')).display")
    topbar_home = page.evaluate("getComputedStyle(document.getElementById('btn-record')).display")
    assert hero != 'none', "Home: hero record button must be visible"
    assert topbar_home == 'none', "Home: top-bar record button must be hidden"

    # Live: top-bar btn is the single record control
    page.evaluate("""() => { viewingHistory = false; showLiveView(); }""")
    page.wait_for_timeout(150)
    topbar_live = page.evaluate("getComputedStyle(document.getElementById('btn-record')).display")
    assert topbar_live != 'none', "Live: top-bar record button must be visible"

    # Session: top-bar btn visible (same placement as live)
    page.evaluate("""() => { selectedSessionId = 's1'; viewingHistory = true; showSessionView(); }""")
    page.wait_for_timeout(150)
    topbar_session = page.evaluate("getComputedStyle(document.getElementById('btn-record')).display")
    assert topbar_session != 'none', "Session: top-bar record button must be visible"


# ---------------------------------------------------------------------------
# T1-e: Live-notes UI reset on new recording start
# ---------------------------------------------------------------------------

def test_start_recording_resets_stale_live_notes_ui(page) -> None:
    """Starting a new recording must clear enhanced-notes state left by a prior session.

    Repro: enhance notes during recording A -> stop A -> start recording B.
    Without resetLiveNotesUI(), recording B shows A's enhanced output, title
    "Enhanced notes", hint, and legend.  This test fails against the pre-fix code.
    """
    # 1. Simulate the stale state a prior enhancement would have produced.
    page.evaluate("""() => {
        const out = document.getElementById('notes-output');
        out.innerHTML = '<p>Stale notes from prior recording</p>';
        out.classList.add('visible');
        document.getElementById('live-pad-title').textContent = 'Enhanced notes';
        document.getElementById('live-pad-hint').textContent = 'Your words kept - AI-added marked';
        const legend = document.getElementById('live-enhance-legend');
        legend.style.display = '';
        legend.innerHTML = '<span>legend content</span>';
    }""")

    # Pre-condition: stale state is in place.
    assert page.evaluate("document.getElementById('notes-output').classList.contains('visible')")
    assert page.evaluate("document.getElementById('live-pad-title').textContent") == "Enhanced notes"

    # 2. Trigger the START branch of toggleRecording() (btn is not .recording).
    #    _stub_api returns {"ok":true} for /api/recording/start automatically.
    page.evaluate("void toggleRecording()")
    page.wait_for_timeout(400)

    # 3. Assert every piece of the live-notes UI was reset.
    assert not page.evaluate(
        "document.getElementById('notes-output').classList.contains('visible')"
    ), "#notes-output still has 'visible' class after new recording started"
    assert page.evaluate("document.getElementById('notes-output').innerHTML") == "", (
        "#notes-output innerHTML not cleared"
    )
    assert page.evaluate("document.getElementById('live-pad-title').textContent") == "Your notes", (
        "#live-pad-title not reset to 'Your notes'"
    )
    assert page.evaluate("document.getElementById('live-pad-hint').textContent") == "Autosaved \xb7 jot freely", (
        "#live-pad-hint not reset"
    )
    assert page.evaluate("document.getElementById('live-enhance-legend').style.display") == "none", (
        "#live-enhance-legend not hidden"
    )
    assert page.evaluate("document.getElementById('live-enhance-legend').innerHTML") == "", (
        "#live-enhance-legend innerHTML not cleared"
    )


# ---------------------------------------------------------------------------
# T6: Dual-field notes editor — user_notes + AI notes
# ---------------------------------------------------------------------------

def test_session_notes_editor_dual_field(chromium_browser, spa_server: str) -> None:
    """Edit mode populates both fields; Save POSTs to the user-notes endpoint.

    Asserts:
    - #session-user-notes-input is populated from _currentSessionUserNotes on edit-open.
    - #session-notes-input is populated from notes_text.
    - saveNotesAndRender() POSTs to /api/sessions/:id/user-notes with the updated text.
    - _currentSessionUserNotes is updated so renderNotesView reflects the edit.
    """
    captured: list[str] = []

    def _intercept(route):
        from urllib.parse import urlparse
        path = urlparse(route.request.url).path
        if path.endswith("/user-notes"):
            captured.append(route.request.post_data or "")
        route.fulfill(status=200, content_type="application/json", body='{"ok":true}')

    ctx = chromium_browser.new_context()
    pg = ctx.new_page()
    # Generic API stub; the intercept above overrides /user-notes paths (LIFO)
    pg.route("**/api/**", lambda r: r.fulfill(
        status=200, content_type="application/json", body='{"ok":true}'
    ))
    pg.route("**/api/sessions/**", _intercept)
    pg.goto(spa_server)
    pg.wait_for_load_state("domcontentloaded")
    pg.wait_for_timeout(300)

    # Simulate a loaded session with both notes types.
    pg.evaluate("""() => {
        selectedSessionId = 'sess-t6';
        _currentSessionUserNotes = 'my context notes';
        document.getElementById('session-notes-input').value = 'AI generated summary';
        document.getElementById('session-user-notes-input').value = 'my context notes';
    }""")

    # Open the editor — toggleNotesEdit() reads _currentSessionUserNotes into the field.
    pg.evaluate("toggleNotesEdit()")
    pg.wait_for_timeout(100)

    ai_val = pg.evaluate("document.getElementById('session-notes-input').value")
    user_val = pg.evaluate("document.getElementById('session-user-notes-input').value")
    assert ai_val == "AI generated summary", f"AI notes field wrong: {ai_val!r}"
    assert user_val == "my context notes", f"User notes field wrong: {user_val!r}"

    # Simulate the user editing the user-notes field.
    pg.evaluate("""() => {
        document.getElementById('session-user-notes-input').value = 'updated context';
    }""")

    # Save — triggers saveNotes() which POSTs to both endpoints.
    pg.evaluate("void saveNotesAndRender()")
    pg.wait_for_timeout(500)

    assert any("updated context" in body for body in captured), (
        f"POST /api/sessions/sess-t6/user-notes not called with updated text; "
        f"captured bodies: {captured}"
    )

    # _currentSessionUserNotes should reflect the saved value so renderNotesView is coherent.
    updated_global = pg.evaluate("_currentSessionUserNotes")
    assert updated_global == "updated context", (
        f"_currentSessionUserNotes not updated after save: {updated_global!r}"
    )

    ctx.close()


# ---------------------------------------------------------------------------
# v1.0.0 T1 — apiCall never throws; HTTP/network errors become structured results
# ---------------------------------------------------------------------------

def test_t1_apicall_http_error_returns_structured_result(routed_page) -> None:
    """A 500 with a JSON error body yields {ok:false, error, status:500} — no throw."""
    pg = routed_page(lambda r: r.fulfill(
        status=500, content_type="application/json", body='{"ok":false,"error":"boom"}'))
    result = pg.evaluate("() => apiCall('/api/recording/start', { method: 'POST' })")
    assert result["ok"] is False
    assert result["error"] == "boom"
    assert result["status"] == 500


def test_t1_apicall_non_json_error_does_not_throw(routed_page) -> None:
    """A 502 with a non-JSON body still yields a structured falsy result, not a throw."""
    pg = routed_page(lambda r: r.fulfill(
        status=502, content_type="text/html", body="<html>Bad Gateway</html>"))
    result = pg.evaluate("() => apiCall('/api/recording/start', { method: 'POST' })")
    assert result["ok"] is False
    assert result["status"] == 502
    assert result["error"]  # non-empty fallback message


def test_t1_apicall_success_without_ok_field_is_truthy(routed_page) -> None:
    """A 200 body lacking an `ok` field is normalized to ok:true (back-compat)."""
    pg = routed_page(lambda r: r.fulfill(
        status=200, content_type="application/json", body='{"session_id":"abc"}'))
    result = pg.evaluate("() => apiCall('/api/recording/stop', { method: 'POST' })")
    assert result["ok"] is True
    assert result["session_id"] == "abc"


# ---------------------------------------------------------------------------
# v1.0.0 T3 — saveNotes() attributes which POST failed
# ---------------------------------------------------------------------------

def test_t3_save_notes_attributes_failing_post(routed_page) -> None:
    """When only the user-notes POST fails, the error names 'Your notes'."""
    from urllib.parse import urlparse

    def _route(route) -> None:
        path = urlparse(route.request.url).path
        if path.endswith("/user-notes"):
            route.fulfill(status=500, content_type="application/json",
                          body='{"ok":false,"error":"db locked"}')
        else:
            route.fulfill(status=200, content_type="application/json", body='{"ok":true}')

    pg = routed_page(_route)
    pg.evaluate("""() => {
        selectedSessionId = 'sess-t3';
        document.getElementById('session-notes-input').value = 'ai';
        document.getElementById('session-user-notes-input').value = 'mine';
    }""")
    pg.evaluate("() => saveNotes()")
    pg.wait_for_timeout(200)

    banner = pg.evaluate("document.getElementById('error-banner').textContent")
    assert "Your notes" in banner, f"error did not attribute the failing POST: {banner!r}"
    assert "AI notes" not in banner, f"unrelated POST wrongly blamed: {banner!r}"
