install:
	uv run python setup_app.py
	cp -r dist/Escriba.app /Applications/

download-model:
	uv run escriba download-model
