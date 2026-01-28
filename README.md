# local-transcriber

Transcriptor local que captura audio del sistema + micrófono (macOS) y procesa audios nuevos en una carpeta. Similar a Notion AI Meeting Notes, proporciona transcripción en tiempo real con baja latencia usando Whisper local.

## Características

- 🎙️ **Captura de audio del sistema y micrófono** usando ScreenCaptureKit (API nativa de macOS)
- ⚡ **Transcripción en tiempo real** con latencia baja (2-5 segundos)
- ⚡ **Optimización CPU** con compute_type int8 para mejor rendimiento
- 🔍 **Auto-detección de dispositivos** (similar a Notion AI)
- 📞 **Detección automática de llamadas** (Zoom, Teams, Meet, etc.)
- 🎧 **Soporte para AirPods** y otros micrófonos externos
- 📁 **Monitoreo de carpetas** para transcripción automática de archivos nuevos
- 📊 **Exportación múltiple** en formatos TXT y JSON estructurado
- ⚙️ **Configuración VAD centralizada** para ajuste fino de detección de voz
- 🔒 **100% local** - no envía datos a servidores externos

## Requisitos (macOS)
- macOS 13.0+ (Ventura o superior) para ScreenCaptureKit
- `ffmpeg` (Homebrew): `brew install ffmpeg`
- Python 3.10+ (recomendado con `uv`)
- Whisper CLI disponible en el PATH (solo para modo `live` legacy)
- `faster-whisper` (instalado automáticamente con dependencias)
- Permisos de Screen Recording (se solicitan automáticamente)

## Instalación

### 1. Instalar dependencias del sistema

```bash
# Instalar ffmpeg (requerido para captura de audio)
brew install ffmpeg
```

### 2. Instalar dependencias Python

```bash
# Crear entorno virtual e instalar el paquete
uv venv
uv pip install -e .
```

### 3. Compilar CLI Swift (para captura de audio del sistema)

```bash
cd swift-audio-capture
swift build -c release
cd ..
```

El ejecutable se encontrará en `swift-audio-capture/.build/release/audio-capture` y será detectado automáticamente.

### 4. Instalar Whisper CLI (solo para modo `live` legacy)

Si planeas usar el modo `live` (legacy), instala Whisper CLI:

```bash
pip install -U openai-whisper
```

Verifica que el comando exista:
```bash
whisper --help
```

**Nota:** Para el modo `live-stream` (recomendado), las dependencias se instalan automáticamente con `uv pip install -e .`.

## Configuración

### Variables de Entorno

El proyecto usa `direnv` y archivos `.env` para configuración. Crea un archivo `.env` basado en `.env.example`:

```bash
cp .env.example .env
```

Si usas `direnv`, crea un `.envrc` con `dotenv`:

```bash
echo 'dotenv' > .envrc
direnv allow
```

### Configuración con TOML (Recomendado)

También puedes usar un archivo `local-transcriber.toml` para configurar defaults.

- Se incluye un ejemplo en `local-transcriber.toml.example`.
- Precedencia:
  - **CLI flags** (para esa ejecución)
  - `local-transcriber.toml`
  - variables de entorno (`.env`)

Opciones:
- `--config /ruta/al/local-transcriber.toml`
- o `LOCAL_TRANSCRIBER_CONFIG=/ruta/al/local-transcriber.toml`
- o `./local-transcriber.toml`

### Configuración Básica

```bash
# Audio
SAMPLE_RATE="16000"              # Sample rate en Hz (8000-48000)
CHANNELS="1"                     # Número de canales (1=mono, 2=estéreo)
AUTO_DETECT_DEVICES=true         # Auto-detección de dispositivos (recomendado)
MIC_ONLY=false                   # true = solo micrófono, false = sistema + micrófono
```

**Nota:** Con `AUTO_DETECT_DEVICES=true` (por defecto), no necesitas configurar `SYSTEM_DEVICE` ni `MIC_DEVICE` manualmente. El sistema detecta automáticamente:
- Audio del sistema usando ScreenCaptureKit (API nativa de macOS)
- Tu micrófono actual (incluyendo AirPods si están conectados)

### Configuración para Modo `live-stream` (Recomendado)

```bash
STREAMING_CHUNK_DURATION=30.0      # Duración de chunks en segundos (0.5+)
STREAMING_MODEL_SIZE=base         # tiny, base, small, medium, large
STREAMING_LANGUAGE=es             # Código ISO 639-1 (es, en, fr, etc.)
STREAMING_DEVICE=auto             # auto|cpu|cuda - solo para faster-whisper (no mps)
STREAMING_BACKEND=faster-whisper  # faster-whisper (CPU optimizado) o openai-whisper (soporta MPS/GPU)
STREAMING_VAD_ENABLED=false       # Voice Activity Detection (true = solo voz, false = todo)
STREAMING_REALTIME_OUTPUT=true    # Mostrar transcripciones en consola en tiempo real
STREAMING_EXPORT_FORMATS=txt,json # Formatos de exportación separados por comas (txt,json)

# VAD (Voice Activity Detection) configuration
VAD_MIN_SILENCE_MS=500            # Duración mínima de silencio en milisegundos
VAD_THRESHOLD=0.3                 # Umbral de sensibilidad (0.0-1.0), valores más bajos detectan más voz
```

### Configuración para Modo `live` (Legacy)

```bash
WHISPER_CMD="whisper --model small --language es --output_format txt --output_dir {output_dir} {input}"
SEGMENT_SECONDS="30"             # Duración de segmentos en segundos
```

### Configuración Manual de Dispositivos (Opcional)

Si prefieres configurar manualmente, desactiva la auto-detección:

```bash
AUTO_DETECT_DEVICES=false
SYSTEM_DEVICE="0"                 # Índice del dispositivo de audio del sistema
MIC_DEVICE="1"                    # Índice del micrófono
```

Para listar dispositivos disponibles:
```bash
local-transcriber list-devices
# o
ffmpeg -f avfoundation -list_devices true -i ""
```

## Arquitectura

### Estructura del Proyecto

```
local-transcriber/
├── src/local_transcriber/
│   ├── cli.py                    # CLI principal (argparse)
│   ├── audio/
│   │   ├── call_detection.py     # Detección de llamadas activas
│   │   ├── device_detection.py   # Auto-detección de dispositivos
│   │   ├── live_capture.py       # Captura en vivo (legacy y streaming)
│   │   └── screen_capture.py     # Captura de audio del sistema (ScreenCaptureKit)
│   ├── transcribe/
│   │   ├── streaming.py          # Transcripción en tiempo real (faster-whisper)
│   │   └── whisper.py             # Transcripción batch (Whisper CLI)
│   └── watch/
│       └── watch_folder.py        # Monitoreo de carpetas para transcripción
├── swift-audio-capture/          # CLI Swift para ScreenCaptureKit
│   ├── Sources/audio-capture/
│   │   ├── AudioCapture.swift    # Captura de audio del sistema
│   │   ├── PCMConverter.swift    # Conversión de formatos
│   │   └── main.swift            # Punto de entrada CLI
│   └── Package.swift
├── pyproject.toml                 # Configuración del proyecto Python
├── .env.example                   # Variables de entorno de ejemplo
└── README.md
```

### Componentes Principales

1. **CLI (`cli.py`)**: Punto de entrada principal con comandos:
   - `live-stream`: Transcripción en tiempo real (recomendado)
   - `live`: Transcripción en segmentos (legacy)
   - `watch`: Monitoreo de carpetas
   - `list-devices`: Lista dispositivos de audio

2. **Captura de Audio**:
   - `screen_capture.py`: Captura audio del sistema usando ScreenCaptureKit (Swift CLI)
   - `live_capture.py`: Orquesta captura de sistema + micrófono y procesamiento
   - `device_detection.py`: Auto-detección de dispositivos (AirPods, micrófonos, etc.)

3. **Transcripción**:
   - `streaming.py`: Usa `faster-whisper` para transcripción en tiempo real
   - `config.py`: Configuración centralizada de VAD
   - `formats.py`: Exportación de transcripciones (TXT, JSON)
   - `whisper.py`: Usa Whisper CLI para transcripción batch

4. **Utilidades**:
   - `call_detection.py`: Detecta llamadas activas (Zoom, Teams, etc.)
   - `watch_folder.py`: Monitorea carpetas y transcribe archivos nuevos

### Flujo de Datos

**Modo Streaming (`live-stream`):**
```
ScreenCaptureKit (Swift) → Audio del Sistema (PCM)
     +
ffmpeg → Micrófono (PCM)
     ↓
Combinación de Audio (numpy)
     ↓
StreamingTranscriber (faster-whisper)
     ↓
Transcripción en tiempo real → Archivo + Consola
```

**Modo Legacy (`live`):**
```
ffmpeg → Sistema + Micrófono → Segmentos WAV (30s)
     ↓
watch_folder detecta archivos nuevos
     ↓
Whisper CLI transcribe cada segmento
     ↓
Transcripciones individuales + combinada
```

## Audio del Sistema

El sistema ahora usa **ScreenCaptureKit** (CLI Swift) para capturar el audio del sistema, igual que Notion AI. **No necesitas BlackHole ni dispositivos virtuales**.

#### Requisitos

1. **macOS 13.0+** (ScreenCaptureKit con audio requiere macOS Ventura)

2. **Permisos de Screen Recording**:
   - La primera vez que ejecutes el script, macOS te pedirá permisos
   - O ve manualmente a: **System Settings > Privacy & Security > Screen Recording**
   - Agrega tu terminal (Terminal, iTerm, etc.)
   - ✅ **No necesitas permisos de accesibilidad** (a diferencia de BlackHole)

3. **Compilar el CLI Swift**:
   ```bash
   cd swift-audio-capture
   swift build -c release
   ```
   El ejecutable se encontrará en `.build/release/audio-capture` y será detectado automáticamente.

#### Ventajas sobre BlackHole

- ✅ **No requiere instalación de software adicional** (BlackHole, Soundflower, etc.)
- ✅ **No requiere configuración manual** (Multi-Output Devices, etc.)
- ✅ **No requiere dependencias Python complejas** (PyObjC) - solo Swift nativo
- ✅ **Captura directa del audio del sistema** (como Notion AI)
- ✅ **Mejor calidad y menor latencia**
- ✅ **Funciona automáticamente** sin configuración

#### Solo Micrófono (Opcional)

Si solo quieres capturar el micrófono y no el audio del sistema:
```bash
export MIC_ONLY=true
```
Esto desactiva la captura de audio del sistema.

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

Esto espera automáticamente a que detecte una llamada activa (Zoom, Teams, Meet, etc.) antes de comenzar.

**Opciones adicionales:**

```bash
local-transcriber live-stream \
  --output-dir transcripts \
  --combined transcripts/live.txt \
  --model-size base \
  --chunk-duration 30.0 \
  --language es \
  --device auto \
  --format txt,json \
  --auto-start \
  --no-realtime-output
```

**Parámetros:**
- `--output-dir`: Directorio donde guardar transcripciones individuales
- `--combined`: Archivo donde escribir transcripción combinada en tiempo real
- `--model-size`: Tamaño del modelo (tiny, base, small, medium, large)
- `--chunk-duration`: Duración de chunks en segundos (default: 30.0)
- `--language`: Idioma (código ISO 639-1: es, en, fr, etc.)
- `--device`: Dispositivo a usar (auto, cpu, mps). `auto` detecta MPS automáticamente en Apple Silicon
- `--format`: Formatos de exportación separados por comas (txt, json). Se exportan al finalizar
- `--auto-start`: Esperar a que se detecte una llamada antes de comenzar
- `--no-realtime-output`: No mostrar transcripciones en consola

### Speaker diarization (pyannote-audio)

Para diarización real (Speaker A/B/etc.) al finalizar la sesión, usa `pyannote-audio` (opcional) y un token de HuggingFace.

- Config:
  - En TOML: `[speaker] mode = "pyannote"`
  - O env: `STREAMING_SPEAKER_MODE=pyannote`
  - Token: `HUGGINGFACE_TOKEN=...`

Notas:
- Esta integración corre diarización **post-run** sobre un WAV de la sesión y luego anota speakers en las exportaciones (JSON/Markdown).

### Modo Legacy (Batch)

Captura en vivo con segmentos de 30s (latencia alta ~30+ segundos):

```bash
local-transcriber live --output-dir transcripts --combined transcripts/combined.txt
```

**Parámetros:**
- `--output-dir`: Directorio donde guardar transcripciones
- `--combined`: Archivo opcional para transcripción combinada

### Transcripción de Carpeta

Monitorea una carpeta y transcribe automáticamente archivos nuevos:

```bash
local-transcriber watch --dir audios --output-dir transcripts --combined transcripts/combined.txt
```

**Parámetros:**
- `--dir`: Directorio a observar (default: `audios`)
- `--output-dir`: Directorio donde guardar transcripciones (default: `transcripts`)
- `--combined`: Archivo opcional para transcripción combinada

**Formatos soportados:** `.wav`, `.mp3`, `.m4a`, `.flac`, `.aac`, `.ogg`, `.mp4`

### Listar Dispositivos de Audio

```bash
local-transcriber list-devices
```

Muestra todos los dispositivos disponibles y los resultados de auto-detección.

## Dependencias

### Runtime Dependencies

- `python-dotenv>=1.0.0`: Carga de variables de entorno
- `openai-whisper>=20250625`: Whisper CLI (solo para modo `live` legacy)
- `faster-whisper>=1.0.0`: Transcripción en tiempo real (modo `live-stream`)
- `numpy>=1.24.0`: Procesamiento de audio
- `sounddevice>=0.4.6`: Interfaz con dispositivos de audio
- `watchdog>=4.0.0`: Monitoreo de carpetas

### Dependencias del Sistema

- **macOS 13.0+** (Ventura o superior) para ScreenCaptureKit
- **ffmpeg**: Instalado vía Homebrew (`brew install ffmpeg`)
- **Swift 5.0+**: Para compilar el CLI de captura de audio (incluido en Xcode Command Line Tools)

### Python

- **Python 3.10+** (recomendado con `uv`)
- Gestión de paquetes: `uv` (recomendado) o `pip`

## Comparación de Modos

### Modo Streaming (`live-stream`) - Recomendado

**Ventajas:**
- ✅ **Latencia baja**: 2-5 segundos (vs 30+ segundos en modo legacy)
- ✅ **Transcripción en tiempo real**: Muestra texto mientras hablas
- ✅ **Sin archivos intermedios**: Procesa chunks directamente en memoria
- ✅ **Similar a Notion AI**: Experiencia de usuario mejorada
- ✅ **Mejor rendimiento**: Usa `faster-whisper` optimizado

**Desventajas:**
- Requiere más memoria RAM (modelo cargado en memoria)
- Dependencia de `faster-whisper` (más pesado que Whisper CLI)

### Modo Legacy (`live`)

**Ventajas:**
- ✅ Compatible con Whisper CLI estándar
- ✅ Puede usar `whisper.cpp` u otros binarios personalizados
- ✅ Menor uso de memoria (procesa archivos individuales)

**Desventajas:**
- ❌ **Latencia alta**: 30+ segundos (espera segmentos completos)
- ❌ Requiere archivos temporales
- ❌ No muestra transcripciones en tiempo real

**Nota:** `WHISPER_CMD` acepta `{input}` y `{output_dir}`. Puedes reemplazarlo por `whisper.cpp` u otro binario si lo necesitas.

### Permisos macOS

- **Permisos de Screen Recording (Requerido)**:
  - Necesario para capturar audio del sistema con ScreenCaptureKit
  - macOS te pedirá automáticamente la primera vez
  - O ve manualmente a: **System Settings > Privacy & Security > Screen Recording**
  - Agrega tu terminal (Terminal, iTerm, etc.)

- **No se requieren**:
  - BlackHole ni dispositivos virtuales
  - Permisos de accesibilidad
  - Configuración manual de dispositivos de audio
  - Dependencias Python complejas (PyObjC) - solo Swift nativo

## Troubleshooting

### Errores Generales

**`command not found: local-transcriber`**
```bash
# Reinstalar el paquete
uv pip install -e .

# O ejecutar directamente
uv run python -m local_transcriber.cli live-stream
```

**Permisos de Screen Recording no otorgados**
- Ve a: **System Settings > Privacy & Security > Screen Recording**
- Agrega tu terminal (Terminal, iTerm, etc.)
- Reinicia la terminal después de otorgar permisos

**CLI Swift no encontrado**
```bash
cd swift-audio-capture
swift build -c release
cd ..
```

### Errores Modo `live-stream`

**`ModuleNotFoundError: faster_whisper`**
```bash
uv pip install -e .
```

**`CUDA out of memory` o memoria insuficiente**
- Usa un modelo más pequeño: `--model-size tiny` o `base`
- Reduce `STREAMING_CHUNK_DURATION` a 1.0-1.5 segundos

**Latencia alta**
- Reduce `STREAMING_CHUNK_DURATION` a 1.0-1.5 segundos
- Usa un modelo más pequeño (`tiny` o `base`)
- Usa GPU con `--device auto` o `--device mps` (Apple Silicon)
- Verifica que no haya otros procesos consumiendo CPU

**Usar GPU en Apple Silicon (MPS) - Limitado**
- ⚠️ **Problema conocido**: `openai-whisper` tiene problemas de estabilidad con MPS que producen errores NaN
- `faster-whisper` (que usa `ctranslate2`) solo soporta `cpu` y `cuda`, no `mps`
- **Recomendación**: Usa `faster-whisper` (backend por defecto) que es más rápido y estable en CPU
- Si quieres intentar `openai-whisper`:
  ```bash
  local-transcriber live-stream --backend openai-whisper
  # Por defecto usará CPU (más estable)
  # Para forzar MPS (puede fallar): export WHISPER_FORCE_MPS=true
  ```
- `openai-whisper` puede ser más rápido en modelos grandes (medium, large) pero tiene problemas con MPS
- `faster-whisper` es más rápido y estable en CPU con todos los modelos
- Requisitos para `openai-whisper`: `pip install openai-whisper torch` (ya incluido en dependencias)

**Error SSL al descargar modelo (openai-whisper)**
Si ves `SSL: CERTIFICATE_VERIFY_FAILED` al usar `openai-whisper`:
```bash
# Solución 1: Instalar certificados de Python
/Applications/Python\ 3.13/Install\ Certificates.command

# Solución 2: Usar directorio de caché personalizado
export WHISPER_CACHE_DIR=~/.cache/whisper
# Descarga el modelo manualmente y colócalo ahí

# Solución 3: Usar faster-whisper (no requiere descarga manual, más estable)
local-transcriber live-stream --backend faster-whisper
```

**Error NaN con openai-whisper y MPS**
Si ves `ValueError: ... nan, nan, nan ...` al usar `openai-whisper`:
- Este es un problema conocido de `openai-whisper` con MPS (GPU)
- **Solución**: El sistema automáticamente usa CPU por defecto
- Si quieres forzar MPS (puede fallar): `export WHISPER_FORCE_MPS=true`
- **Recomendación**: Usa `faster-whisper` que es más rápido y estable:
  ```bash
  local-transcriber live-stream --backend faster-whisper
  ```

**No se detecta audio del sistema**
- Verifica permisos de Screen Recording
- Verifica que el CLI Swift esté compilado: `swift-audio-capture/.build/release/audio-capture`
- Prueba con `MIC_ONLY=true` para solo usar micrófono

### Errores Modo `live` (Legacy)

**`FileNotFoundError: whisper`**
```bash
pip install -U openai-whisper
# Verifica que esté en PATH
whisper --help
```

**Segmentos no se procesan**
- Verifica que `WHISPER_CMD` esté correctamente configurado
- Verifica permisos de escritura en `output_dir`
- Revisa logs para errores de Whisper CLI

### Problemas de Audio

**No se detecta micrófono**
```bash
# Listar dispositivos disponibles
local-transcriber list-devices

# Configurar manualmente si es necesario
export AUTO_DETECT_DEVICES=false
export MIC_DEVICE="1"  # Usar índice correcto
```

**AirPods no se detectan automáticamente**
- Verifica que estén conectados y activos
- Usa `local-transcriber list-devices` para verificar
- Si no aparecen, configura manualmente con `MIC_DEVICE`

**Audio del sistema no se captura**
- Verifica permisos de Screen Recording
- Verifica que el CLI Swift esté compilado y funcionando
- Prueba ejecutando manualmente: `swift-audio-capture/.build/release/audio-capture --list`

## Desarrollo

### Estructura del Código

- **CLI**: `src/local_transcriber/cli.py`
- **Captura de Audio**: `src/local_transcriber/audio/`
- **Transcripción**: `src/local_transcriber/transcribe/`
- **Monitoreo**: `src/local_transcriber/watch/`
- **Swift CLI**: `swift-audio-capture/`

### Ejecutar Tests

```bash
# Test del CLI Swift
cd swift-audio-capture
swift test
```

### Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## Licencia

Ver `LICENSE` para más información.
