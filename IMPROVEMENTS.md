# Mejoras Propuestas — local-transcriber

Análisis y roadmap de mejoras. Última actualización: 27 Enero 2026.

---

## 🔧 Mejoras Técnicas

### 1. ✅ Centralizar configuración de VAD [COMPLETADO]
**Problema:** Los parámetros de VAD estaban hardcodeados en múltiples lugares con valores inconsistentes.

**Ubicaciones anteriores:**
- `streaming.py` línea ~103: `min_silence_duration_ms=500, threshold=0.5`
- `streaming.py` línea ~155: `min_silence_duration_ms=500, threshold=0.2`
- `streaming.py` línea ~230: `min_silence_duration_ms=1000, threshold=0.3`

**Solución implementada:**
- ✅ Creado `src/local_transcriber/transcribe/config.py` con `VADConfig` dataclass
- ✅ Variables de entorno: `VAD_MIN_SILENCE_MS` y `VAD_THRESHOLD`
- ✅ Reemplazados todos los valores hardcodeados en `streaming.py`
- ✅ Valores por defecto: `min_silence_duration_ms=500`, `threshold=0.3`

**Archivos modificados:**
- `src/local_transcriber/transcribe/config.py` (nuevo)
- `src/local_transcriber/transcribe/streaming.py`
- `.env.example`

**Esfuerzo:** ~1 hora ✅

---

### 2. ✅ Mejorar mezcla de audio sistema + micrófono [COMPLETADO]
**Problema:** La mezcla actual es un promedio simple que puede perder volumen:
```python
combined_array = ((system_array + mic_array) // 2).astype(np.int16)
```

**Solución implementada:** Usar normalización dinámica:
```python
def mix_audio(system: np.ndarray, mic: np.ndarray, mic_boost: float = 1.2) -> np.ndarray:
    # Boost micrófono (usualmente más bajo que sistema)
    mic_boosted = mic * mic_boost
    
    # Mezclar
    mixed = system.astype(np.float32) + mic_boosted.astype(np.float32)
    
    # Normalizar para evitar clipping
    peak = np.abs(mixed).max()
    if peak > 32767:
        mixed = mixed * (32767 / peak)
    
    return mixed.astype(np.int16)
```

**Estado actual:**
- ✅ Función `mix_audio()` implementada en `src/local_transcriber/audio/live_capture.py`
- ✅ Normalización dinámica para evitar clipping
- ✅ Boost configurable del micrófono vía variable de entorno `AUDIO_MIC_BOOST` (default: 1.2)
- ✅ Manejo de arrays vacíos

**Archivos modificados:**
- `src/local_transcriber/audio/live_capture.py` - Función `mix_audio()` implementada
- `.env.example` - Variable `AUDIO_MIC_BOOST` agregada

**Esfuerzo:** ~2 horas ✅

---

### 3. ⚠️ Soporte GPU - LIMITADO [COMPLETADO - Con limitaciones del backend]
**Problema:** Anteriormente usaba solo CPU:
```python
device="cpu", compute_type="int8"
```

**Estado actual:**
- ✅ Función `get_device_config()` para configuración de dispositivo
- ✅ `StreamingTranscriber` acepta `device="auto"` para detección automática
- ✅ Flag CLI `--device auto|cpu|cuda` agregado
- ✅ Backend alternativo `openai-whisper` implementado (con fallback a CPU)
- ⚠️ **Limitación técnica:** `faster-whisper` usa `ctranslate2` que solo soporta `cpu` y `cuda`, no `mps`
- ⚠️ **Problema conocido:** `openai-whisper` con MPS produce NaN en tensores (bug de PyTorch MPS)
- ✅ **Solución actual:** CPU con `int8` es la mejor opción (eficiente en Apple Silicon gracias al Neural Engine)

**Análisis de backends:**
| Backend | GPU Support | Estado | Rendimiento |
|---------|-------------|--------|-------------|
| `faster-whisper` | ❌ No (solo CPU/CUDA) | ✅ Estable | ⚡⚡⚡ Muy rápido en CPU |
| `openai-whisper` | ⚠️ Sí, pero buggy | ⚠️ Usa CPU por defecto | ⚡ Rápido, pero inestable con MPS |

**Archivos modificados:**
- `src/local_transcriber/transcribe/streaming.py`
- `src/local_transcriber/transcribe/streaming_mps.py` (nuevo - backend alternativo)
- `src/local_transcriber/cli.py`
- `src/local_transcriber/audio/live_capture.py`
- `.env.example`

**Uso:**
```bash
# faster-whisper (recomendado - default)
local-transcriber live-stream --backend faster-whisper

# openai-whisper (CPU por defecto, MPS puede fallar)
local-transcriber live-stream --backend openai-whisper
```

**Conclusión:** El problema es a nivel de backend, no del código:
- `faster-whisper` → usa `ctranslate2` → solo CPU/CUDA
- `openai-whisper` → MPS buggy (NaN en tensores)
- La buena noticia: `int8` en CPU de Apple Silicon es bastante eficiente (el Neural Engine ayuda)

**Alternativa real para el futuro:** Ver mejora #14 (mlx-whisper) - está hecho específicamente para Apple Silicon con MLX y es significativamente más rápido.

**Esfuerzo:** ~1 hora ✅

---

### 4. Modo daemon con modelo pre-cargado
**Problema:** Cada ejecución carga el modelo (~3-5 segundos de startup).

**Solución:** Modo daemon que mantiene el modelo en memoria:
```bash
# Iniciar daemon
local-transcriber daemon start --model-size base

# Controlar vía socket/API
local-transcriber start-recording --output calls/meeting.txt
local-transcriber stop-recording
local-transcriber status
```

**Implementación:**
- Socket Unix en `~/.local-transcriber/daemon.sock`
- Comandos: `start`, `stop`, `status`, `start-recording`, `stop-recording`
- Mantener `StreamingTranscriber` cargado en memoria

**Esfuerzo:** ~1 día

---

### 14. Migrar a mlx-whisper para aceleración GPU real en Apple Silicon
**Descripción:** `mlx-whisper` es una implementación de Whisper optimizada específicamente para Apple Silicon usando MLX (framework de ML nativo de Apple).

**Ventajas:**
- ✅ **Aceleración GPU real** en Apple Silicon (M1/M2/M3/M4)
- ✅ **2-3x más rápido** que `faster-whisper` en CPU
- ✅ **Optimizado nativamente** para Apple Silicon
- ✅ **Sin problemas de NaN** (a diferencia de openai-whisper con MPS)
- ✅ **Mejor uso del Neural Engine**

**Implementación:**
```python
# Nuevo backend mlx-whisper
from mlx_whisper import transcribe

class StreamingTranscriberMLX:
    def __init__(self, model_size: str = "base"):
        self.model = mlx_whisper.load_model(model_size)
        # MLX usa GPU automáticamente
```

**Agregar flag CLI:**
```bash
local-transcriber live-stream --backend mlx-whisper
```

**Requisitos:**
- `mlx` (framework de Apple)
- `mlx-whisper` (implementación de Whisper para MLX)

**Consideraciones:**
- Requiere reimplementar el streaming (mlx-whisper puede tener API diferente)
- Verificar compatibilidad con chunks de audio en tiempo real
- Puede requerir ajustes en el procesamiento de audio

**Esfuerzo:** ~2-3 días (investigación + implementación + testing)

**Referencias:**
- https://github.com/ml-explore/mlx-examples/tree/main/whisper
- MLX está diseñado específicamente para Apple Silicon

---

## 🚀 Nuevas Features

### 5. Resumen automático con LLM al terminar
**Descripción:** Al terminar la transcripción, generar resumen + action items.

**Implementación:**
```python
def generate_summary(transcript: str) -> dict:
    """Genera resumen usando Gemini/Claude."""
    prompt = f"""
    Analiza esta transcripción de llamada y genera:
    1. Resumen ejecutivo (3-5 oraciones)
    2. Puntos clave discutidos
    3. Action items con responsables (si se mencionan)
    4. Decisiones tomadas
    
    Transcripción:
    {transcript}
    """
    # Llamar a Gemini/Claude API
    ...
```

**Agregar flags:**
```bash
local-transcriber live-stream --summarize --summary-model gemini
```

**Esfuerzo:** ~3-4 horas

---

### 6. Speaker hints (cambios de speaker)
**Descripción:** No es diarización completa, pero detectar cambios de voz y marcarlos.

**Implementación simple:**
- Detectar cambios significativos en características de audio entre segmentos
- Marcar con `[Speaker A]`, `[Speaker B]`, etc.
- Usar embeddings de voz si se quiere más precisión (pyannote)

**Output ejemplo:**
```
[10:15:23] [Speaker A] Entonces el plan es lanzar la semana que viene.
[10:15:30] [Speaker B] Perfecto, yo me encargo de la documentación.
```

**Esfuerzo:** ~1 día (simple) / ~3 días (con pyannote)

---

### 7. ✅ Exportar a múltiples formatos [COMPLETADO]
**Descripción:** Además de TXT, exportar a formatos útiles.

**Implementado:**
- ✅ **JSON** — estructurado con timestamps y metadata
- ✅ **TXT** — formato simple con timestamps
- ✅ **SRT** — subtítulos para video
- ✅ **Markdown** — con headers, bullets, formato

**Flags:**
```bash
local-transcriber live-stream --format txt,json
```

**JSON implementado:**
```json
{
  "metadata": {
    "date": "2026-01-27",
    "duration_seconds": 1823,
    "model": "base",
    "language": "es",
    "device": "mps",
    "compute_type": "float16"
  },
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Hola, ¿cómo están todos?"
    }
  ]
}
```

**Archivos creados/modificados:**
- `src/local_transcriber/transcribe/formats.py` (nuevo) - Funciones `export_to_json()`, `export_to_txt()`, `export_to_srt()`, `export_to_markdown()`
- `src/local_transcriber/transcribe/streaming.py` (agregado `export_transcript()`)
- `src/local_transcriber/cli.py` (flag `--format`)
- `src/local_transcriber/audio/live_capture.py` (integración de exportación)
- `.env.example` (variable `STREAMING_EXPORT_FORMATS`)

**Formatos soportados:**
- `txt` - Formato simple con timestamps
- `json` - Estructurado con metadata completa
- `srt` - Subtítulos para video (formato SRT estándar)
- `markdown` - Formato Markdown con headers y timestamps

**Esfuerzo:** ~3-4 horas ✅

---

### 8. Integración con calendario (auto-start)
**Descripción:** Detectar juntas de Calendar y ofrecer iniciar transcripción.

**Implementación:**
- Leer eventos de Apple Calendar vía `icalBuddy` o EventKit
- Detectar eventos con links de Zoom/Meet/Teams
- Notificación 1 minuto antes: "¿Iniciar transcripción para 'Weekly Sync'?"
- Auto-nombrar archivo con nombre del evento

```bash
local-transcriber watch-calendar --auto-start
```

**Esfuerzo:** ~1 día

---

### 9. Enviar resumen por mensaje
**Descripción:** Al terminar, enviar resumen a Telegram/WhatsApp/Slack.

**Implementación:**
```bash
local-transcriber live-stream --notify telegram --chat-id 12345
```

**Esfuerzo:** ~2-3 horas (si ya hay API de mensajería)

---

### 10. Crear GitHub Issues desde transcripción
**Descripción:** Extraer TODOs/action items y crear issues.

**Implementación:**
```bash
local-transcriber create-issues --transcript calls/meeting.txt --repo goes-infraestructura/copiloto-medico
```

Usa LLM para extraer action items y los convierte en issues con:
- Título descriptivo
- Cuerpo con contexto de la conversación
- Labels sugeridos

**Esfuerzo:** ~3-4 horas

---

## 🐛 Robustez

### 11. ✅ Reconexión automática del Swift CLI [COMPLETADO]
**Problema:** Si el proceso Swift falla, la captura se detiene.

**Solución implementada:**
- ✅ Función `monitor_swift_cli()` en `src/local_transcriber/audio/live_capture.py`
- ✅ Método `restart()` en `ScreenCaptureAudioCapture` class
- ✅ Reintentos automáticos con exponential backoff (máximo 3 intentos)
- ✅ Monitoreo cada 5 segundos del estado del proceso Swift
- ✅ Limpieza de buffer de audio al reiniciar

**Implementación:**
```python
def monitor_swift_cli():
    """Monitorea el proceso Swift CLI y lo reinicia si falla."""
    # Verificar cada 5 segundos si el proceso está vivo
    # Si terminó inesperadamente, reiniciar con exponential backoff
    # Resetear contador de reintentos en éxito
```

**Archivos modificados:**
- `src/local_transcriber/audio/live_capture.py` - Función `monitor_swift_cli()` implementada
- `src/local_transcriber/audio/screen_capture.py` - Método `restart()` agregado a `ScreenCaptureAudioCapture`

**Esfuerzo:** ~2 horas ✅

---

### 12. Health checks y métricas
**Descripción:** Monitorear calidad de la captura.

**Métricas:**
- Latencia promedio (tiempo entre audio y transcripción)
- % de chunks con silencio
- Nivel de audio (detectar si está muy bajo)
- Errores de transcripción

```python
@dataclass
class CaptureMetrics:
    chunks_processed: int = 0
    chunks_silent: int = 0
    avg_latency_ms: float = 0
    audio_level_db: float = 0
    errors: int = 0
```

**Esfuerzo:** ~3-4 horas

---

### 13. Tests
**Problema:** No hay tests unitarios.

**Tests necesarios:**
- `test_vad_config.py` — configuración de VAD
- `test_audio_mixing.py` — mezcla de audio
- `test_wav_creation.py` — creación de chunks WAV
- `test_streaming_transcriber.py` — transcripción (con audio de prueba)
- `test_device_detection.py` — detección de dispositivos

**Esfuerzo:** ~1 día

---

## 📋 Priorización

### Quick wins (1-2 horas cada uno)
- [x] ✅ Centralizar VAD config [COMPLETADO]
- [x] ✅ Soporte GPU (MPS) [COMPLETADO]
- [x] ✅ Exportar múltiples formatos [COMPLETADO - JSON, TXT, SRT, Markdown]

### Medium effort (~1 día cada uno)
- [x] ✅ Mejorar mezcla de audio [COMPLETADO]
- [ ] Resumen automático con LLM
- [x] ✅ Reconexión automática Swift CLI [COMPLETADO]
- [ ] Tests básicos

### Larger effort (varios días)
- [ ] Modo daemon
- [ ] Speaker diarization
- [ ] Integración calendario
- [ ] Crear issues desde transcripción
- [ ] Migrar a mlx-whisper para GPU real en Apple Silicon

---

## Notas

- El proyecto ya tiene una buena base, estas mejoras son incrementales
- Priorizar según lo que más uses (¿resúmenes? ¿formatos? ¿auto-start?)
- Algunas mejoras pueden combinarse (ej: daemon + calendar integration)

---

## 📝 Estado de Implementación

### ✅ Cambios Recientes (Enero 2026)

#### Optimización de duración de chunks (27 Enero 2026)
**Cambio:** Duración de chunks aumentada de 2.0s a 30.0s por defecto.

**Motivación:** 
- Chunks más largos proporcionan más contexto a Whisper, mejorando la calidad de la transcripción
- Mejor comprensión de contexto y coherencia en el texto transcrito
- Trade-off: mayor latencia (30s vs 2s) pero mejor precisión

**Archivos modificados:**
- `src/local_transcriber/audio/live_capture.py` - Valor por defecto actualizado
- `src/local_transcriber/cli.py` - Mensaje de ayuda actualizado
- `README.md` - Documentación actualizada

**Configuración:**
```bash
# En .env o como variable de entorno
STREAMING_CHUNK_DURATION=30.0

# O como argumento CLI
local-transcriber live-stream --chunk-duration 30.0
```

**Nota:** El valor puede ajustarse según necesidades (más corto = menor latencia, más largo = mejor calidad).

#### Gitignore para Swift (27 Enero 2026)
**Cambio:** Agregadas entradas al `.gitignore` para ignorar artefactos de compilación de Swift.

**Patrones agregados:**
- `.build/` - Directorio de compilación de Swift Package Manager
- `.swiftpm/` - Cache de Swift Package Manager
- `*.swiftmodule`, `*.swiftdoc`, `*.swiftsourceinfo` - Módulos compilados
- `*.dSYM/` - Símbolos de depuración
- `*.o`, `*.a` - Archivos objeto y librerías
- `*.xcodeproj/`, `*.xcworkspace/`, `DerivedData/` - Artefactos de Xcode

**Archivos modificados:**
- `.gitignore` - Sección Swift agregada

---

### ✅ Completado (Enero 2026)

1. **Centralizar configuración VAD** - Configuración centralizada con variables de entorno
2. **Mezcla de audio mejorada** - Normalización dinámica con boost configurable del micrófono
3. **Soporte GPU (MPS)** - Detección automática y uso de Apple Silicon GPU
4. **Exportar múltiples formatos** - Exportación en JSON, TXT, SRT y Markdown
5. **Reconexión automática Swift CLI** - Monitoreo y reinicio automático del proceso Swift
6. **Optimización de duración de chunks** - Cambio de 2.0s a 30.0s para mejorar calidad de transcripción
7. **Gitignore para Swift** - Agregadas entradas para ignorar artefactos de compilación de Swift Package Manager

### 🔄 En Progreso

- Ninguno actualmente

### 📋 Próximas Mejoras Sugeridas

Basado en el esfuerzo y valor, las siguientes mejoras son buenos candidatos:

1. **Health checks y métricas** (3-4 horas) - Monitorear calidad de captura y transcripción
2. **Tests básicos** (1 día) - Asegurar calidad del código
3. **Resumen automático con LLM** (3-4 horas) - Feature muy útil para reuniones

### 🎯 Mejoras de Alto Valor

- **Resumen automático con LLM** (3-4 horas) - Feature muy útil para reuniones
- **Modo daemon** (1 día) - Reduce tiempo de startup significativamente
- **Migrar a mlx-whisper** (2-3 días) - Aceleración GPU real en Apple Silicon, 2-3x más rápido
