# local-transcriber
Transcriptor local que captura audio del sistema + micrófono (macOS) y procesa
audios nuevos en una carpeta.

## Requisitos (macOS)
- Instalar BlackHole (2ch) como driver virtual de audio.
- `ffmpeg` (Homebrew): `brew install ffmpeg`
- Python 3.10+ (recomendado con `uv`).
- Whisper CLI disponible en el PATH (solo para modo `live` legacy).
- `faster-whisper` (instalado automáticamente con dependencias).

## Instalación
```bash
uv venv
uv pip install -e .
```

Para el modo `live-stream` (recomendado), las dependencias se instalan automáticamente.

Para el modo `live` (legacy), instala Whisper CLI:
```bash
pip install -U openai-whisper
```
Verifica que el comando exista:
```bash
whisper --help
```

## Configuración
`direnv` ya carga tu `.env`. Si usas `direnv`, crea un `.envrc` con `dotenv` y
usa `.env.example` como base.

### Configuración básica
```
SAMPLE_RATE="16000"
CHANNELS="1"
AUTO_DETECT_DEVICES=true  # Detecta dispositivos automáticamente (recomendado)
```

**Nota:** Con `AUTO_DETECT_DEVICES=true` (por defecto), no necesitas configurar `SYSTEM_DEVICE` ni `MIC_DEVICE` manualmente. El sistema detecta automáticamente:
- BlackHole para audio del sistema (si está instalado)
- Tu micrófono actual (incluyendo AirPods si están conectados)

Si prefieres configuración manual, desactiva la auto-detección:
```
AUTO_DETECT_DEVICES=false
SYSTEM_DEVICE="0"  # Solo necesario si AUTO_DETECT_DEVICES=false
MIC_DEVICE="1"     # Solo necesario si AUTO_DETECT_DEVICES=false
```

### Configuración para modo `live-stream` (recomendado)
```
STREAMING_CHUNK_DURATION=2.0      # Duración de chunks en segundos
STREAMING_MODEL_SIZE=base         # tiny, base, small, medium, large
STREAMING_LANGUAGE=es              # Código ISO 639-1 (es, en, etc.)
STREAMING_VAD_ENABLED=true        # Voice Activity Detection
STREAMING_REALTIME_OUTPUT=true     # Mostrar transcripciones en consola
```

### Configuración para modo `live` (legacy)
```
WHISPER_CMD="whisper --model small --language es --output_format txt --output_dir {output_dir} {input}"
SEGMENT_SECONDS="30"
```

Activa `direnv` si es la primera vez:
```bash
direnv allow
```

### Auto-detección de Dispositivos (Recomendado)

Por defecto, el sistema detecta automáticamente los dispositivos de audio, similar a Notion AI:
- **Audio del sistema**: Detecta BlackHole automáticamente (si está instalado)
- **Micrófono**: Detecta automáticamente tu micrófono actual, incluyendo AirPods si están conectados

**Ventajas:**
- No necesitas configurar índices manualmente
- Funciona automáticamente cuando conectas/desconectas AirPods
- Similar a la experiencia de Notion AI

### Configuración Manual (Opcional)

Si prefieres configurar manualmente, desactiva la auto-detección:
```bash
AUTO_DETECT_DEVICES=false
SYSTEM_DEVICE="0"
MIC_DEVICE="1"
```

Para listar dispositivos disponibles:
```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

### Configuración de BlackHole (Solo si usas audio del sistema)

Para capturar audio del sistema (Zoom, Teams, etc.):
- Instala BlackHole: `brew install blackhole-2ch`
- En **Audio MIDI Setup**, configura la salida del sistema a **BlackHole 2ch**
  o crea un **Multi-Output Device** para escuchar y capturar a la vez.
- El sistema lo detectará automáticamente con `AUTO_DETECT_DEVICES=true`

## Uso

### Modo Streaming (Recomendado) - Similar a Notion AI
Transcripción en tiempo real con latencia baja (2-5 segundos):
```bash
local-transcriber live-stream --output-dir transcripts --combined transcripts/live.txt
```

**Auto-inicio cuando detecta una llamada** (como Notion AI):
```bash
local-transcriber live-stream --auto-start
```
Esto espera automáticamente a que detecte una llamada activa (Zoom, Teams, etc.) antes de comenzar.

Opciones adicionales:
```bash
local-transcriber live-stream \
  --output-dir transcripts \
  --combined transcripts/live.txt \
  --model-size base \
  --chunk-duration 2.0 \
  --language es \
  --auto-start
```

### Modo Legacy (Batch)
Captura en vivo con segmentos de 30s (latencia alta):
```bash
local-transcriber live --output-dir transcripts --combined transcripts/combined.txt
```

### Transcripción de carpeta
```bash
local-transcriber watch --dir audios --output-dir transcripts --combined transcripts/combined.txt
```

## Notas

### Modo Streaming (`live-stream`)
- **Latencia baja**: 2-5 segundos (vs 30+ segundos en modo legacy)
- **Transcripción en tiempo real**: Muestra texto mientras hablas
- **Sin archivos intermedios**: Procesa chunks directamente en memoria
- **Similar a Notion AI**: Experiencia de usuario mejorada
- Usa `faster-whisper` para mejor rendimiento

### Modo Legacy (`live`)
- `WHISPER_CMD` acepta `{input}` y `{output_dir}`. Puedes reemplazarlo por
  `whisper.cpp` u otro binario si lo necesitas.
- Guarda segmentos temporales en una carpeta temporal y los transcribe en orden.
- Útil si necesitas compatibilidad con Whisper CLI o procesamiento por lotes.

### Permisos macOS
- **No requiere permisos adicionales**: Usa BlackHole como ya está configurado
- La configuración actual de BlackHole sigue siendo necesaria para capturar audio del sistema
- No se requieren permisos de Screen Recording o Microphone adicionales

## Troubleshooting

### Errores generales
- `command not found: local-transcriber`: ejecuta `uv pip install -e .` o usa
  `uv run python -m local_transcriber.cli ...`.
- No aparece BlackHole: reinstala el driver y reinicia apps/`ffmpeg`.

### Errores modo `live-stream`
- `ModuleNotFoundError: faster_whisper`: ejecuta `uv pip install -e .` para instalar dependencias.
- `CUDA out of memory`: usa `--model-size tiny` o `base` en lugar de modelos más grandes.
- Latencia alta: reduce `STREAMING_CHUNK_DURATION` a 1.0-1.5 segundos.

### Errores modo `live` (legacy)
- `FileNotFoundError: whisper`: instala `openai-whisper` o ajusta `WHISPER_CMD`.
