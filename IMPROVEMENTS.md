# Mejoras Propuestas — local-transcriber

Análisis y roadmap de mejoras. Enero 2026.

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

### 2. Mejorar mezcla de audio sistema + micrófono
**Problema:** La mezcla actual es un promedio simple que puede perder volumen:
```python
combined_array = ((system_array + mic_array) // 2).astype(np.int16)
```

**Solución:** Usar normalización dinámica:
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

**Esfuerzo:** ~2 horas

---

### 3. ✅ Soporte para GPU (MPS en Apple Silicon) [COMPLETADO]
**Problema:** Anteriormente usaba solo CPU:
```python
device="cpu", compute_type="int8"
```

**Solución implementada:**
- ✅ Función `get_device_config()` que detecta MPS automáticamente
- ✅ `StreamingTranscriber` acepta `device="auto"` para detección automática
- ✅ Flag CLI `--device auto|cpu|mps` agregado
- ✅ Si MPS disponible: `device="mps"`, `compute_type="float16"`
- ✅ Si no: `device="cpu"`, `compute_type="int8"` (fallback)

**Archivos modificados:**
- `src/local_transcriber/transcribe/streaming.py`
- `src/local_transcriber/cli.py`
- `src/local_transcriber/audio/live_capture.py`
- `.env.example`

**Uso:**
```bash
local-transcriber live-stream --device auto  # Detecta automáticamente
local-transcriber live-stream --device mps   # Fuerza MPS
local-transcriber live-stream --device cpu   # Fuerza CPU
```

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

### 7. ✅ Exportar a múltiples formatos [COMPLETADO - Parcial]
**Descripción:** Además de TXT, exportar a formatos útiles.

**Implementado:**
- ✅ **JSON** — estructurado con timestamps y metadata
- ✅ **TXT** — formato simple con timestamps
- ⏳ **SRT** — subtítulos para video (pendiente)
- ⏳ **Markdown** — con headers, bullets, formato (pendiente)

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
- `src/local_transcriber/transcribe/formats.py` (nuevo)
- `src/local_transcriber/transcribe/streaming.py` (agregado `export_transcript()`)
- `src/local_transcriber/cli.py` (flag `--format`)
- `src/local_transcriber/audio/live_capture.py` (integración de exportación)
- `.env.example` (variable `STREAMING_EXPORT_FORMATS`)

**Pendiente:**
- SRT export (subtítulos)
- Markdown export (formato con headers)

**Esfuerzo:** ~3-4 horas (JSON/TXT completado, SRT/Markdown pendiente)

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

### 11. Reconexión automática del Swift CLI
**Problema:** Si el proceso Swift falla, la captura se detiene.

**Solución:**
```python
def _read_audio_stream(self):
    max_retries = 3
    retry_count = 0
    
    while not self.stop_event.is_set():
        try:
            # ... leer audio ...
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Max retries reached, stopping")
                break
            logger.warning(f"Swift CLI failed, retrying ({retry_count}/{max_retries})")
            self._restart_swift_process()
```

**Esfuerzo:** ~2 horas

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
- [x] ✅ Exportar JSON estructurado [COMPLETADO - Parcial: JSON/TXT, pendiente SRT/Markdown]

### Medium effort (~1 día cada uno)
- [ ] Mejorar mezcla de audio
- [ ] Resumen automático con LLM
- [ ] Reconexión automática Swift CLI
- [ ] Tests básicos

### Larger effort (varios días)
- [ ] Modo daemon
- [ ] Speaker diarization
- [ ] Integración calendario
- [ ] Crear issues desde transcripción

---

## Notas

- El proyecto ya tiene una buena base, estas mejoras son incrementales
- Priorizar según lo que más uses (¿resúmenes? ¿formatos? ¿auto-start?)
- Algunas mejoras pueden combinarse (ej: daemon + calendar integration)

---

## 📝 Estado de Implementación

### ✅ Completado (Enero 2026)

1. **Centralizar configuración VAD** - Configuración centralizada con variables de entorno
2. **Soporte GPU (MPS)** - Detección automática y uso de Apple Silicon GPU
3. **Exportar JSON estructurado** - Exportación en formato JSON con metadata completa

### 🔄 En Progreso

- Ninguno actualmente

### 📋 Próximas Mejoras Sugeridas

Basado en el esfuerzo y valor, las siguientes mejoras son buenos candidatos:

1. **Mejorar mezcla de audio** (2 horas) - Normalización dinámica para mejor calidad
2. **Reconexión automática Swift CLI** (2 horas) - Mayor robustez
3. **Completar formatos de exportación** (2-3 horas) - Agregar SRT y Markdown
4. **Tests básicos** (1 día) - Asegurar calidad del código

### 🎯 Mejoras de Alto Valor

- **Resumen automático con LLM** (3-4 horas) - Feature muy útil para reuniones
- **Modo daemon** (1 día) - Reduce tiempo de startup significativamente
