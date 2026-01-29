# Mejoras Propuestas — local-transcriber

Análisis y roadmap de mejoras. Última actualización: 28 Enero 2026.

**Estado:** 14 mejoras completadas en Enero 2026. Proyecto con base sólida y features avanzadas implementadas.

---

## 🔥 Prioridad Alta (Nuevas)

### 15. Speaker Diarization Real (Pyannote)
**Problema:** Actualmente el `speaker/detection.py` es muy básico. No sabemos quién está hablando.
**Solución:** Integrar `pyannote-audio` para identificar locutores (`Speaker A`, `Speaker B`).
**Por qué:** Es la "killer feature" para competir con Otter.ai.
**Esfuerzo:** Alto (~3-4 días). Requiere manejar modelos de HuggingFace.

### 16. Refactor de CLI a Typer/Click
**Problema:** `cli.py` tiene 300+ líneas de `argparse` y lógica mezclada. Es difícil de mantener.
**Solución:** Migrar a **Typer**.
- Menos código boilerplate.
- Help bonito y con colores automático.
- Validación de tipos gratis.
**Esfuerzo:** Medio (~1 día).

### 17. Configuración Centralizada (TOML)
**Problema:** Demasiadas variables de entorno (`.env`) y flags.
**Solución:** Soportar un archivo de configuración `local-transcriber.toml`.
```toml
[model]
size = "base"
backend = "faster-whisper"

[audio]
mic_boost = 1.2
```
**Esfuerzo:** Bajo (~3-4 horas).

---

## 🔧 Mejoras Técnicas (Roadmap Previo)

### 1. ✅ Centralizar configuración de VAD [COMPLETADO]
*Implementado en Enero 2026.*
- Variables de entorno: `VAD_MIN_SILENCE_MS` y `VAD_THRESHOLD`.
- Archivos: `src/local_transcriber/transcribe/config.py`.

### 2. ✅ Mejorar mezcla de audio sistema + micrófono [COMPLETADO]
*Implementado en Enero 2026.*
- Normalización dinámica y `AUDIO_MIC_BOOST`.
- Archivos: `src/local_transcriber/audio/live_capture.py`.

### 3. ⚠️ Soporte GPU - LIMITADO [COMPLETADO]
*Implementado en Enero 2026.*
- Flags `--device auto|cpu|cuda`.
- **Nota:** `faster-whisper` solo CPU/CUDA. `openai-whisper` tiene bugs en MPS.
- **Siguiente paso:** Ver mejora #14 (mlx-whisper).

### 4. ✅ Modo daemon con modelo pre-cargado [COMPLETADO]
*Implementado en Enero 2026.*
- Socket Unix en `~/.local-transcriber/daemon.sock`
- Comandos: `daemon start|stop|status`, `start-recording`, `stop-recording`
- Mantiene `StreamingTranscriber` cargado en memoria
- Archivos: `src/local_transcriber/daemon/` (server.py, client.py)
- CLI: `local-transcriber daemon start --model-size base`

### 14. ✅ Migrar a mlx-whisper (Apple Silicon) [COMPLETADO]
*Implementado en Enero 2026.*
- Backend `mlx-whisper` implementado en `streaming_mlx.py`
- Aceleración GPU real en Apple Silicon (M1/M2/M3/M4)
- Uso: `--backend mlx-whisper`
- Archivos: `src/local_transcriber/transcribe/streaming_mlx.py`
- **Nota:** Requiere `pip install mlx-whisper`

---

## 🚀 Nuevas Features (Ideas)

### 18. WebSocket Server
**Idea:** `local-transcriber serve --port 8765`.
**Uso:** Integrar con otras apps (OBS, webs locales) que necesiten subtítulos en vivo.

### 19. Hotkey Global
**Idea:** `Cmd+Shift+T` para iniciar/parar grabación sin tocar la terminal.
**Req:** Probablemente requiera una mini-app en menubar o integración con Shortcuts.

### 20. Buffer de Retroalimentación (Ring Buffer)
**Idea:** "Always on" mode que guarda los últimos 30s.
**Uso:** "¡Eso que dijiste fue importante!" -> Iniciar grabación y ya tener los 30s previos.

### 21. Notificaciones Nativas
**Idea:** Usar `osascript` o librerías nativas para notificar: "Transcripción guardada en..."
**Estado:** ✅ Telegram implementado, falta macOS nativo.

### 6. ✅ Speaker Hints (Detección básica) [COMPLETADO]
*Implementado en Enero 2026.*
- Detección simple de cambios de voz basada en características de audio
- Análisis espectral (centroide, rolloff, RMS, zero-crossing rate)
- Marca segmentos con `[Speaker A]`, `[Speaker B]`
- Uso: `--speaker-detection` o `STREAMING_SPEAKER_DETECTION=true`
- Configuración: `SPEAKER_DETECTION_THRESHOLD=0.3`
- Archivos: `src/local_transcriber/speaker/` (detection.py)
- **Nota:** Implementación básica. Para diarización completa ver #15 (Pyannote).

### 8. ✅ Integración con Calendario [COMPLETADO]
*Implementado en Enero 2026.*
- Observa Apple Calendar para detectar reuniones próximas
- Detecta eventos con links de Zoom/Meet/Teams
- Auto-start opcional para transcripciones
- Uso: `watch-calendar --auto-start`
- Archivos: `src/local_transcriber/calendar/` (apple_calendar.py)

### 9. ✅ Enviar Resumen por Mensaje [COMPLETADO]
*Implementado en Enero 2026.*
- Integración con Telegram para enviar resúmenes
- Formato Markdown con resumen ejecutivo, puntos clave y action items
- Uso: `STREAMING_NOTIFY=true`, `STREAMING_NOTIFY_PLATFORM=telegram`
- Variables: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- Archivos: `src/local_transcriber/notify/` (telegram.py)

### 10. ✅ Crear GitHub Issues desde Transcripción [COMPLETADO]
*Implementado en Enero 2026.*
- Extrae action items de transcripciones usando LLM
- Crea issues automáticamente en GitHub
- Incluye contexto de la conversación
- Uso: `create-issues --transcript <file> --repo owner/repo --model gemini`
- Variables: `GITHUB_TOKEN`
- Archivos: `src/local_transcriber/integrations/` (github.py)

### 5. ✅ Resumen automático con LLM [COMPLETADO]
*Implementado en Enero 2026.*
- Soporte para Gemini y Claude APIs
- Genera resumen estructurado (ejecutivo, puntos clave, action items, decisiones)
- Guarda en `_summary.json` al finalizar
- Uso: `--summarize --summary-model gemini`
- Variables: `STREAMING_SUMMARIZE`, `STREAMING_SUMMARY_MODEL`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`
- Archivos: `src/local_transcriber/summarize/` (llm_summary.py)

---

## 🐛 Robustez y Calidad

### 13. ✅ Tests Básicos [COMPLETADO]
*Implementado en Enero 2026.*
- `test_vad_config.py` ✅
- `test_audio_mixing.py` ✅
- `test_device_detection.py` ✅ (nuevo)
- `test_streaming_transcriber.py` ✅ (nuevo)
- `test_formats.py` ✅
- `test_wav_creation.py` ✅
- **Total:** 36 tests pasando
- **Archivos:** `tests/test_*.py`

### 12. ✅ Health checks y métricas [COMPLETADO]
*Implementado en Enero 2026.*
- Clase `CaptureMetrics` con tracking completo
- Métricas: latencia promedio/min/max, % chunks silenciosos, nivel de audio (dB), errores
- Integrado en `StreamingTranscriber` y `live_capture.py`
- Uso: `--metrics` o `STREAMING_SHOW_METRICS=true`
- Archivos: `src/local_transcriber/transcribe/metrics.py`

---

## 📋 Priorización Actualizada (Enero 2026)

### Quick Wins (Low Hanging Fruit)
1. **Config TOML** (#17) - Limpia mucho el uso.
2. **Notificaciones Nativas** (#21) - Feedback visual inmediato.
3. **Refactor CLI** (#16) - Paga deuda técnica.

### ✅ Completado (Enero 2026)
1. ✅ **Tests** (#13) - 36 tests implementados y pasando
2. ✅ **Resumen LLM** (#5) - Gemini y Claude support
3. ✅ **Modo Daemon** (#4) - Socket Unix server implementado
4. ✅ **MLX Whisper** (#14) - Backend implementado
5. ✅ **Health Checks** (#12) - Métricas completas
6. ✅ **Speaker Hints** (#6) - Detección básica implementada
7. ✅ **Calendario** (#8) - Integración básica
8. ✅ **Notificaciones** (#9) - Telegram implementado
9. ✅ **GitHub Issues** (#10) - Creación automática

### Próximos Pasos (Prioridad)
1. **Speaker Diarization Real** (#15) - Pyannote para diarización completa
2. **Config TOML** (#17) - Configuración centralizada
3. **Refactor CLI** (#16) - Migrar a Typer
4. **Notificaciones Nativas macOS** (#21) - Completar notificaciones

---

## 📝 Estado de Implementación

### ✅ Completado Recientemente (Enero 2026)

#### Mejoras Técnicas
1. **Centralizar VAD config** - Variables de entorno centralizadas
2. **Mezcla de audio mejorada** - Normalización dinámica con mic_boost
3. **Soporte GPU** - faster-whisper (CPU/CUDA), openai-whisper (MPS con limitaciones), mlx-whisper (Apple Silicon)
4. **Exportar múltiples formatos** - JSON, TXT, SRT, Markdown
5. **Reconexión automática Swift CLI** - Monitoreo y reinicio automático

#### Nuevas Features
6. **Health checks y métricas** - Tracking completo de latencia, audio levels, errores
7. **Tests básicos** - 36 tests pasando (device detection, streaming transcriber, formats, etc.)
8. **Resumen automático con LLM** - Gemini y Claude support
9. **Modo daemon** - Modelo pre-cargado en memoria, socket Unix
10. **Speaker hints** - Detección básica de cambios de voz
11. **Integración calendario** - Auto-start para reuniones
12. **Notificaciones Telegram** - Envío automático de resúmenes
13. **GitHub Issues** - Creación automática desde action items
14. **mlx-whisper backend** - Aceleración GPU real en Apple Silicon

#### Mejoras de Robustez
15. **Optimización de chunks (30s)** - Mejor calidad de transcripción
16. **Fix streaming shutdown** - Transcriber inicializado correctamente
17. **Debug mejorado** - Stderr tail + comando en crash

---

## 📊 Resumen de Implementación (Enero 2026)

### Estadísticas
- **Mejoras completadas:** 14
- **Tests implementados:** 36 (todos pasando)
- **Nuevos módulos creados:** 7
  - `summarize/` - Resúmenes con LLM
  - `daemon/` - Modo daemon con modelo pre-cargado
  - `speaker/` - Detección de cambios de voz
  - `calendar/` - Integración con Apple Calendar
  - `notify/` - Notificaciones (Telegram)
  - `integrations/` - Integraciones externas (GitHub)
  - `transcribe/metrics.py` - Health checks y métricas

### Archivos Principales Modificados
- `src/local_transcriber/transcribe/streaming.py` - Métricas y speaker detection
- `src/local_transcriber/transcribe/streaming_mlx.py` - Backend MLX (nuevo)
- `src/local_transcriber/audio/live_capture.py` - Integración de todas las features
- `src/local_transcriber/cli.py` - Nuevos comandos y flags
- `.env.example` - Variables de entorno documentadas
- `tests/` - Suite completa de tests

### Comandos Nuevos Disponibles
```bash
# Daemon mode
local-transcriber daemon start --model-size base
local-transcriber daemon status
local-transcriber daemon start-recording --output-dir transcripts
local-transcriber daemon stop-recording

# Features avanzadas
local-transcriber live-stream --summarize --summary-model gemini
local-transcriber live-stream --speaker-detection
local-transcriber live-stream --metrics
local-transcriber live-stream --backend mlx-whisper

# Integraciones
local-transcriber watch-calendar --auto-start
local-transcriber create-issues --transcript file.txt --repo owner/repo
```

### Próximos Pasos Recomendados
1. **Speaker Diarization Real** (#15) - Mejorar detección básica con Pyannote
2. **Config TOML** (#17) - Simplificar configuración
3. **Refactor CLI** (#16) - Migrar a Typer para mejor UX
4. **Notificaciones macOS** (#21) - Completar notificaciones nativas
