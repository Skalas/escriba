# Escriba

## Installing as macOS app

Build and install the .app bundle:

```bash
uv run python setup_app.py
cp -r dist/Escriba.app /Applications/
```

This creates a lightweight wrapper that launches the Python app via `uv run escriba app`. Do NOT use `pip install -e .` — the app runs through uv from the project directory.

## Running in development

```bash
uv run escriba app
```

## Key commands

- `uv run escriba app` — start menu bar app + HTTP server
- `uv run python setup_app.py` — build .app bundle into `dist/`
