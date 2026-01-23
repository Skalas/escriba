# local-transcriber
Transcriptor local que captura audio del sistema + micrófono (macOS) y procesa
audios nuevos en una carpeta.

## Requisitos (macOS)
- Instalar BlackHole (2ch) como driver virtual de audio.
- `ffmpeg` (Homebrew): `brew install ffmpeg`
- Python 3.10+ (recomendado con `uv`).
- Whisper CLI disponible en el PATH.

## Instalación
```bash
uv venv
uv pip install -e .
```

Instala Whisper (opción recomendada):
```bash
pip install -U openai-whisper
```
Verifica que el comando exista:
```bash
whisper --help
```

## Configuración
`direnv` ya carga tu `.env`. Si usas `direnv`, crea un `.envrc` con `dotenv` y
usa `.env.example` como base. Asegúrate de tener algo como:
```
WHISPER_CMD="whisper --model small --language es --output_format txt --output_dir {output_dir} {input}"
SYSTEM_DEVICE="0"
MIC_DEVICE="1"
SAMPLE_RATE="16000"
CHANNELS="1"
SEGMENT_SECONDS="30"
```

Activa `direnv` si es la primera vez:
```bash
direnv allow
```

### Dispositivos de audio (macOS)
Para listar dispositivos de audio en macOS:
```bash
ffmpeg -f avfoundation -list_devices true -i ""
```
Usa los índices resultantes en `SYSTEM_DEVICE` y `MIC_DEVICE`. Con BlackHole, el
audio del sistema debe estar ruteado a ese dispositivo virtual. Es normal que
`ffmpeg` termine con un error al listar dispositivos (solo se usa para enumerar).

### Configuración de BlackHole
- En **Audio MIDI Setup**, configura la salida del sistema a **BlackHole 2ch**
  o crea un **Multi-Output Device** para escuchar y capturar a la vez.
- Verifica que BlackHole aparezca en `ffmpeg -list_devices`.

## Uso
Captura en vivo (sistema + mic):
```bash
local-transcriber live --output-dir transcripts --combined transcripts/combined.txt
```

Transcripción de carpeta:
```bash
local-transcriber watch --dir audios --output-dir transcripts --combined transcripts/combined.txt
```

## Notas
- `WHISPER_CMD` acepta `{input}` y `{output_dir}`. Puedes reemplazarlo por
  `whisper.cpp` u otro binario si lo necesitas.
- El modo `live` guarda segmentos temporales en una carpeta temporal y los
  transcribe en orden.

## Troubleshooting
- `command not found: local-transcriber`: ejecuta `uv pip install -e .` o usa
  `uv run python -m local_transcriber.cli ...`.
- `FileNotFoundError: whisper`: instala `openai-whisper` o ajusta `WHISPER_CMD`.
- No aparece BlackHole: reinstala el driver y reinicia apps/`ffmpeg`.
